"""The chunked linear + cross-entropy path, with an explicit backward pass.

The paper's formulation is

    loss = mean_i [ logsumexp_j(C_j^T E_i) - C_{x_i}^T E_i ]               (Eq. 4)

where ``C`` is the ``(V, D)`` classifier and ``E`` the ``(N, D)`` hidden
states. The expensive term is the log-sum-exp over the vocabulary. Here the
vocabulary is cut into chunks, ``logits_c = E @ C_c^T`` is materialised for one
chunk at a time, and each chunk is reduced to a ``(max, sum-of-exp)`` pair
before the next chunk is touched. The largest per-chunk array is ``N x chunk``
instead of ``N x |V|``.

Backward pass
-------------
Autodiff through the Python chunk loop also keeps memory proportional to the
chunk, but it must keep the chunk logits alive for the backward sweep. Instead
the whole operator is wrapped in ``mx.custom_function`` with an explicit
``vjp`` that *recomputes* the logits chunk by chunk, exactly as the paper's
backward kernel does (Section 4.3): the softmax weights are recovered from the
stored log-sum-exp with

    S_i = exp(logits_i - LSE_i)                (no renormalisation needed)

and the gradients are accumulated chunk by chunk,

    dL/dE   += sum_c (S_c * coeff) @ C_c
    dL/dC_c  = (S_c * coeff)^T @ E

so the only per-chunk state is a ``N x chunk`` logits block plus the ``N x D``
running ``dL/dE`` and the ``V x D`` head gradient (which is the lower bound the
paper quotes: any implementation has to write it). The ``vjp`` also implements
the paper's block-level gradient skipping.
"""

from __future__ import annotations

import mlx.core as mx

from .ce import (
    IGNORE_INDEX,
    OnlineLSE,
    cross_entropy_from_lse,
    naive_cross_entropy,
    online_update,
)
from .filter import gradient_filter_report, softmax_is_negligible
from .kernels import row_stats

# Largest vocabulary block the Metal reduction can take in one launch.
MAX_FUSED_BLOCK = 1024

REDUCTIONS = ("fused", "mlx", "inline")


def _vocab_size(head: mx.array) -> int:
    return int(head.shape[0])


def _n_chunks(vocab: int, chunk_size: int) -> int:
    return max(1, -(-vocab // chunk_size))


def _check_inputs(hidden: mx.array, head: mx.array, targets: mx.array) -> None:
    if hidden.ndim != 2 or head.ndim != 2:
        raise ValueError(f"hidden and head must be 2D, got {hidden.shape} and {head.shape}")
    if hidden.shape[1] != head.shape[1]:
        raise ValueError(f"dimension mismatch: hidden {hidden.shape} vs head {head.shape}")
    if targets.shape != (hidden.shape[0],):
        raise ValueError(f"targets must be ({hidden.shape[0]},), got {targets.shape}")


def naive_linear_cross_entropy(
    hidden: mx.array,
    head: mx.array,
    targets: mx.array,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[mx.array, mx.array]:
    """Reference path: build the whole ``(N, V)`` logits matrix, then reduce it.

    This is what Cut Cross-Entropy replaces. Peak memory here is proportional to
    ``N * |V|`` because both the logits and (during the backward pass) the
    softmax over them have to exist in full.
    """
    _check_inputs(hidden, head, targets)
    logits = hidden @ head.T
    return naive_cross_entropy(logits, targets, ignore_index=ignore_index)


def _target_logits_chunked(
    hidden: mx.array, head: mx.array, targets: mx.array, chunk_size: int
) -> mx.array:
    """The ground-truth logit for every token, touching one chunk at a time."""
    n = hidden.shape[0]
    vocab = _vocab_size(head)
    safe = mx.where(targets == IGNORE_INDEX, mx.zeros_like(targets), targets)
    safe = mx.clip(safe, 0, vocab - 1)
    out = mx.zeros((n,), mx.float32)
    start = 0
    while start < vocab:
        stop = min(start + chunk_size, vocab)
        local = safe - start
        hit = (local >= 0) & (local < stop - start)
        chunk = hidden @ head[start:stop].T
        picked = mx.take_along_axis(chunk, mx.clip(local, 0, stop - start - 1)[:, None], axis=1)[:, 0]
        out = mx.where(hit, picked, out)
        start = stop
    return out


def _chunk_summaries(
    hidden: mx.array, head: mx.array, chunk_size: int, reduction: str
) -> list[tuple[mx.array, mx.array]]:
    """Per-chunk ``(max, sum-of-exp)``, one ``(N,)`` pair per vocabulary chunk.

    Chunks are ragged: the last one just stops at the vocabulary boundary.
    Nothing is padded, because a padded ``-inf`` column would make an
    all-padding chunk summarise to ``(-inf, 0)`` and the online merge would
    then have to evaluate ``exp(-inf - -inf)``.
    """
    vocab = _vocab_size(head)
    out: list[tuple[mx.array, mx.array]] = []
    start = 0
    while start < vocab:
        stop = min(start + chunk_size, vocab)
        block = (hidden @ head[start:stop].T).astype(mx.float32)
        if reduction == "fused":
            if stop - start > MAX_FUSED_BLOCK:
                raise ValueError(
                    f"reduction='fused' handles at most {MAX_FUSED_BLOCK} columns per "
                    f"chunk, got {stop - start}; use reduction='mlx' or a smaller chunk_size"
                )
            cmax, csum = row_stats(block)
        else:
            cmax = mx.max(block, axis=-1)
            csum = mx.sum(mx.exp(block - cmax[:, None]), axis=-1)
        out.append((cmax, csum))
        start = stop
    return out


def _chunked_lse(
    hidden: mx.array, head: mx.array, chunk_size: int, reduction: str
) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
    """``logsumexp`` over the vocabulary, one chunk at a time.

    ``reduction`` selects how a chunk is summarised. All three options produce
    the same summaries; ``"inline"`` differs only in that it merges each chunk
    into the running accumulator before computing the next one, which is the
    literal reading of the paper's Algorithm 2 and the lowest-memory variant.

    Returns the log-sum-exp and the per-chunk summaries, which the backward
    pass reuses to decide which chunks it can skip.
    """
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    summaries = _chunk_summaries(hidden, head, chunk_size, reduction)
    state = OnlineLSE(summaries[0][0], summaries[0][1])
    for cmax, csum in summaries[1:]:
        state = online_update(state, cmax, csum)
    return state.max + mx.log(state.sum_exp), summaries


def _count_valid(targets: mx.array, ignore_index: int) -> mx.array:
    """Number of non-ignored tokens, as a float32 scalar (or 1 when there are none)."""
    return mx.maximum(mx.sum(mx.stop_gradient(targets != ignore_index).astype(mx.float32)), 1.0)


def _mean_from_per_token(per_token: mx.array, targets: mx.array, ignore_index: int) -> mx.array:
    """Masked mean of the per-token losses, computed outside the operator."""
    valid = mx.stop_gradient(targets != ignore_index).astype(mx.float32)
    return mx.sum(per_token * valid) / _count_valid(targets, ignore_index)


def _per_token_loss(
    hidden: mx.array,
    head: mx.array,
    targets: mx.array,
    chunk_size: int,
    reduction: str,
    ignore_index: int,
) -> mx.array:
    """``lse_i - logit_{x_i}`` for every token, with ignored rows zeroed.

    The log-sum-exp stays *inside* the operator: it is the shared factor behind
    both terms of Equation 4, and handing it out as a separate output would make
    the cotangent of that output carry gradient the ``vjp`` below is already
    accounting for.
    """
    lse, _ = _chunked_lse(hidden, head, chunk_size, reduction)
    target_logits = _target_logits_chunked(hidden, head, targets, chunk_size)
    ignore = targets == ignore_index
    target_logits = mx.where(ignore, mx.zeros_like(target_logits), target_logits)
    _, per_token = cross_entropy_from_lse(lse, target_logits, ignore)
    # A token whose entire vocabulary is -inf has no distribution to score. The
    # naive path produces 0 - (-inf) = +inf there; report 0 instead, which keeps
    # an all -inf classifier matrix from poisoning the loss.
    finite = mx.isfinite(lse)
    return mx.where(finite, per_token, mx.zeros_like(per_token))


@mx.custom_function
def _chunked_linear_ce_op(
    hidden: mx.array,
    head: mx.array,
    targets: mx.array,
    chunk_size: int,
    reduction: str,
    ignore_index: int,
    threshold: mx.array,
) -> mx.array:
    """Per-token loss of the chunked path. Differentiate the caller's mean of it.

    The output is deliberately the *per-token* loss and not the mean: the mean,
    the ignore mask and any reweighting the user wants then happen in ordinary
    MLX code outside the operator, which is the property the paper highlights
    (CCE has separate forward and backward stages, so the loss can be
    transformed by the user).
    """
    return _per_token_loss(hidden, head, targets, chunk_size, reduction, ignore_index)


@_chunked_linear_ce_op.vjp
def _chunked_linear_ce_vjp(primals, cotangents, outputs):
    hidden, head, targets, chunk_size, reduction, ignore_index, threshold = primals
    cots = tuple(cotangents) if isinstance(cotangents, (list, tuple)) else (cotangents,)
    cot = cots[0] if len(cots) > 0 else None
    if cot is None:
        cot = mx.zeros((hidden.shape[0],), mx.float32)
    cot = mx.asarray(cot).astype(mx.float32)

    n, d = hidden.shape
    vocab = _vocab_size(head)
    eps = float(threshold)
    ignore = targets == ignore_index
    lse, summaries = _chunked_lse(hidden, head, chunk_size, reduction)
    # Rows with no finite logit have no softmax; use 0 so exp() stays finite
    # there. Their per-token loss is pinned to 0 above, so this is consistent.
    softmax_lse = mx.where(mx.isfinite(lse), lse, mx.zeros_like(lse))

    # The cotangent arriving here is already the mean over non-ignored tokens,
    # so the only extra weighting is the ignore mask itself.
    coeff = mx.where(ignore, mx.zeros((n,), mx.float32), cot)

    # d(loss_i)/d(logit_i) is -coeff_i at the ground-truth token and +coeff_i *
    # softmax_i everywhere else, so the gathered term of Equation 4 contributes
    # -coeff_i * head[target_i] to the hidden gradient and -coeff_i * hidden_i to
    # head[target_i]. The latter is a one-hot scatter into the vocabulary axis.
    safe_targets = mx.where(ignore, mx.zeros_like(targets), targets)
    safe_targets = mx.clip(safe_targets, 0, vocab - 1)
    grad_hidden = -(coeff[:, None] * head[safe_targets])
    grad_head = mx.zeros((vocab, d), mx.float32)

    skipped: list[bool] = []
    start = 0
    while start < vocab:
        stop = min(start + chunk_size, vocab)
        cols = head[start:stop]
        chunk = (hidden @ cols.T).astype(mx.float32)
        # S_i = softmax(logits_i), recovered from the stored log-sum-exp without
        # recomputing the normaliser.
        s = mx.exp(chunk - softmax_lse[:, None])
        if eps > 0.0:
            # Block-level skip, as in the paper: a chunk all of whose softmax
            # weights are below the threshold contributes nothing to either
            # gradient. This is a conservative test - it can only fire when the
            # chunk's largest weight is already below eps.
            negligible = softmax_is_negligible(s, eps)
            if bool(mx.all(negligible)):
                skipped.append(True)
                start = stop
                continue
            s = mx.where(negligible, mx.zeros_like(s), s)
        skipped.append(False)
        weighted = s * coeff[:, None]
        # In-place accumulation, so the backward graph does not keep one buffer
        # per chunk alive at once.
        grad_hidden += weighted @ cols
        # Each chunk only ever writes its own column range of the head gradient,
        # so no full-size softmax matrix is needed to scatter into it.
        grad_head[start:stop] = weighted.T @ hidden
        # ... plus the one-hot scatter of the gathered term, restricted to the
        # targets that fall inside this chunk. Building the one-hot for the chunk
        # costs one (N, chunk) boolean, not one (N, V).
        in_chunk = (~ignore) & (safe_targets >= start) & (safe_targets < stop)
        if bool(mx.any(in_chunk)):
            local = safe_targets - start
            one_hot = mx.arange(stop - start)[None, :] == local[:, None]
            scatter = (one_hot.astype(mx.float32) * coeff[:, None]) * in_chunk[:, None]
            grad_head[start:stop] -= scatter.T @ hidden
        start = stop

    global _LAST_FILTER_REPORT
    _LAST_FILTER_REPORT = gradient_filter_report(mx.array(skipped, mx.bool_), n)

    return (
        grad_hidden.astype(hidden.dtype),
        grad_head.astype(head.dtype),
        None,
        None,
        None,
        None,
        None,
    )


def chunked_linear_cross_entropy(
    hidden: mx.array,
    head: mx.array,
    targets: mx.array,
    *,
    chunk_size: int = 512,
    reduction: str = "fused",
    ignore_index: int = IGNORE_INDEX,
    grad_threshold: float = 0.0,
) -> tuple[mx.array, mx.array]:
    """Cross-entropy over ``hidden @ head.T`` without materialising the logits.

    Parameters
    ----------
    hidden:
        ``(N, D)`` hidden states (the backbone output for every token).
    head:
        ``(V, D)`` classifier / unembedding matrix.
    targets:
        ``(N,)`` integer targets; entries equal to ``ignore_index`` contribute
        nothing to the loss, its gradient, or the normalisation.
    chunk_size:
        Number of vocabulary entries per chunk. Peak memory grows with this.
    reduction:
        ``"fused"`` (a Metal kernel per chunk, ``chunk_size <= 1024``),
        ``"mlx"`` (the same reduction with ``mx.max`` / ``mx.sum``), or
        ``"inline"`` (a serial chunk loop that merges each chunk into the
        running log-sum-exp before computing the next one).
    grad_threshold:
        Sparse-gradient skip, mirroring the paper's gradient filtering. A chunk
        whose softmax weights are *all* strictly below this threshold is not
        accumulated into either gradient. ``0.0`` disables the skip.
    """
    _check_inputs(hidden, head, targets)
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
    if reduction == "fused" and chunk_size > MAX_FUSED_BLOCK:
        raise ValueError(
            f"reduction='fused' handles at most {MAX_FUSED_BLOCK} columns per chunk, "
            f"got {chunk_size}; use reduction='mlx' or a smaller chunk_size"
        )
    per_token = _chunked_linear_ce_op(
        hidden,
        head,
        targets,
        int(chunk_size),
        str(reduction),
        int(ignore_index),
        mx.array(float(grad_threshold), mx.float32),
    )
    return _mean_from_per_token(per_token, targets, ignore_index), per_token


def linear_cross_entropy(
    hidden: mx.array,
    head: mx.array,
    targets: mx.array,
    *,
    chunk_size: int | None = None,
    reduction: str = "fused",
    ignore_index: int = IGNORE_INDEX,
    grad_threshold: float = 0.0,
) -> tuple[mx.array, mx.array]:
    """One interface for both paths.

    ``chunk_size=None`` runs the naive reference and materialises the full
    ``(N, V)`` logits matrix; any positive integer runs the chunked path.
    """
    if chunk_size is None:
        return naive_linear_cross_entropy(hidden, head, targets, ignore_index=ignore_index)
    return chunked_linear_cross_entropy(
        hidden,
        head,
        targets,
        chunk_size=chunk_size,
        reduction=reduction,
        ignore_index=ignore_index,
        grad_threshold=grad_threshold,
    )


# Written by the ``vjp`` above. MLX's ``custom_function`` objects do not accept
# extra attributes, so the statistics of the most recent backward pass live
# here; the experiment and the tests read them through ``last_filter_report``.
_LAST_FILTER_REPORT: dict[str, float] = {}


def last_filter_report() -> dict[str, float]:
    """Block-level skip statistics from the most recent chunked backward pass."""
    return dict(_LAST_FILTER_REPORT)
