"""Numerically stable cross-entropy in log space, and the reference (naive) path.

The cross-entropy for a single token is

    loss_i = logsumexp_j(C_j^T E_i) - C_{x_i}^T E_i                       (Eq. 4)

so it needs exactly two things from the logits: the log-sum-exp over the whole
vocabulary, and the logit of the ground-truth token. Everything in this module
is organised around producing those two scalars with as little materialised
state as possible.

``naive_cross_entropy`` takes an already-materialised ``(N, V)`` logits matrix
and is the correctness reference. ``chunked_cross_entropy`` takes a *callable*
that produces one vocabulary chunk of logits at a time, and reduces each chunk
into a running (max, sum-of-exp) pair as it goes, so the largest array it ever
holds is one chunk.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import mlx.core as mx

IGNORE_INDEX = -100


class OnlineLSE(NamedTuple):
    """A running estimate of ``logsumexp`` over the vocabulary, per token.

    ``max`` and ``sum_exp`` satisfy ``logsumexp = max + log(sum_exp)``. Keeping
    the pair rather than the log-sum-exp itself is what makes the accumulator
    exact: chunks are folded in with ``exp(x - max)`` and never with
    ``exp(x - logsumexp)``.
    """

    max: mx.array
    sum_exp: mx.array


def online_update(running: OnlineLSE, chunk_max: mx.array, chunk_sum: mx.array) -> OnlineLSE:
    """Fold one chunk's ``(max, sum-exp)`` summary into the running accumulator.

    This is the online-softmax merge: rescale whichever of the two partial
    results is further from the new maximum. Note that it is a merge of two
    summaries, not a scan of raw logits, so it is exact up to floating point
    regardless of how many chunks it is applied to.
    """
    new_max = mx.maximum(running.max, chunk_max)
    # exp(x - new_max) is 0 when x == -inf and new_max is finite, and is
    # exp(nan) == nan when both are -inf, matching a single-pass reduction.
    running_scale = mx.exp(running.max - new_max)
    chunk_scale = mx.exp(chunk_max - new_max)
    new_sum = running.sum_exp * running_scale + chunk_sum * chunk_scale
    return OnlineLSE(max=new_max, sum_exp=new_sum)


def logsumexp_from_lse(state: OnlineLSE) -> mx.array:
    """``max + log(sum_exp)``, with the empty-vocabulary case folded to ``-inf``."""
    return state.max + mx.log(state.sum_exp)


def cross_entropy_from_lse(
    lse: mx.array,
    target_logits: mx.array,
    ignore_mask: mx.array,
) -> tuple[mx.array, mx.array]:
    """``loss = mean_i(lse_i - logit_{x_i})`` over non-ignored tokens.

    Returns ``(loss, per_token)`` where ``per_token`` is
    ``lse_i - logit_{x_i}`` with ignored entries zeroed, so the caller can
    inspect individual rows without recomputing anything.
    """
    per_token = mx.where(ignore_mask, mx.zeros_like(lse), lse - target_logits)
    n = mx.sum(~ignore_mask).astype(mx.float32)
    total = mx.sum(per_token)
    loss = mx.where(n > 0, total / mx.maximum(n, 1.0), mx.zeros_like(total))
    return loss, per_token


def _ignore_mask(targets: mx.array) -> mx.array:
    return targets == IGNORE_INDEX


def _clip_targets(targets: mx.array, vocab: int) -> mx.array:
    """Make ignored targets safe to gather with, without changing their mask."""
    return mx.where(targets == IGNORE_INDEX, mx.zeros_like(targets), targets) % vocab


def naive_cross_entropy(
    logits: mx.array,
    targets: mx.array,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[mx.array, mx.array]:
    """Cross-entropy from a materialised ``(N, V)`` logits matrix.

    This is the formulation Cut Cross-Entropy removes: ``mx.logsumexp`` over the
    last axis touches every element of the ``(N, V)`` matrix, and because
    autodiff keeps ``exp(logits)`` alive for the backward pass the peak memory
    is a multiple of ``N * V``, not ``N * V``.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be (N, V), got shape {logits.shape}")
    if targets.shape != (logits.shape[0],):
        raise ValueError(f"targets must be ({logits.shape[0]},), got {targets.shape}")
    ignore = _ignore_mask(targets) if ignore_index is not None else mx.zeros(targets.shape, mx.bool_)
    safe = _clip_targets(targets, logits.shape[1]) if ignore_index is not None else targets
    target_logits = mx.take_along_axis(logits, safe[:, None], axis=1)[:, 0]
    lse = mx.logsumexp(logits, axis=-1)
    return cross_entropy_from_lse(lse, target_logits, ignore)


def chunked_cross_entropy(
    logits_chunk: Callable[[int, int], mx.array],
    targets: mx.array,
    *,
    chunk_size: int,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[mx.array, mx.array]:
    """Cross-entropy that never holds more than one vocabulary chunk of logits.

    ``logits_chunk(start, stop)`` must return the ``(N, stop - start)`` slice of
    the logits matrix. The reduction mirrors Algorithm 2 of the paper: every
    chunk is summarised to ``(max, sum-exp)`` and merged with the online
    log-sum-exp rule, and the ground-truth logit is picked up from whichever
    chunk contains it.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    n = targets.shape[0]
    ignore = _ignore_mask(targets) if ignore_index is not None else mx.zeros(targets.shape, mx.bool_)

    state = OnlineLSE(max=mx.full((n,), -mx.inf, mx.float32), sum_exp=mx.zeros((n,), mx.float32))
    target_logits = mx.zeros((n,), mx.float32)
    vocab = 0
    start = 0
    while True:
        stop = start + chunk_size
        chunk = logits_chunk(start, stop)
        if chunk.shape[1] == 0:
            break
        chunk = chunk.astype(mx.float32)
        v = chunk.shape[1]
        state = online_update(state, mx.max(chunk, axis=1), mx.sum(mx.exp(chunk - mx.max(chunk, axis=1)[:, None]), axis=1))
        # The ground-truth token lives in exactly one chunk; every other chunk
        # contributes nothing to this term.
        local = targets - start
        hit = (~ignore) & (local >= 0) & (local < v)
        picked = mx.take_along_axis(chunk, mx.clip(local, 0, v - 1)[:, None], axis=1)[:, 0]
        target_logits = mx.where(hit, picked, target_logits)
        vocab += v
        start = stop
        if v < chunk_size:
            break
    if vocab == 0:
        raise ValueError("logits_chunk produced no columns")
    return cross_entropy_from_lse(logsumexp_from_lse(state), target_logits, ignore)
