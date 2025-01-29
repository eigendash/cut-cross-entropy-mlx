"""The invariants that matter for the chunked path.

The central one is gradient equality with the naive, matrix-materialising path:
the loss is easy to get right, the backward pass is where a Cut Cross-Entropy
implementation goes wrong (it did here - the first version of the custom ``vjp``
forgot the gathered ground-truth term).
"""

import math

import numpy as np
import pytest

import mlx.core as mx

from cce.filter import BFLOAT16_EPSILON
from cce.linear_ce import (
    REDUCTIONS,
    chunked_linear_cross_entropy,
    last_filter_report,
    linear_cross_entropy,
    naive_linear_cross_entropy,
)

CHUNK_SIZES = (1, 3, 7, 64, 512)
BIG_CHUNK = 10_000


def _problem(seed=0, n=7, d=16, v=200, dtype=mx.float32):
    rng = np.random.default_rng(seed)
    hidden = mx.array(rng.normal(size=(n, d)).astype(np.float32) * 0.7, dtype)
    head = mx.array(rng.normal(size=(v, d)).astype(np.float32) * 0.7, dtype)
    targets = mx.array(rng.integers(0, v, size=n).astype(np.int32))
    return hidden, head, targets


def _grad(fn, hidden, head):
    hidden = mx.array(np.asarray(hidden, dtype=np.float32))
    head = mx.array(np.asarray(head, dtype=np.float32))
    value, (g_hidden, g_head) = mx.value_and_grad(fn, argnums=(0, 1))(hidden, head)
    mx.eval(value, g_hidden, g_head)
    return float(value), np.array(g_hidden), np.array(g_head)


# --------------------------------------------------------------------------- loss


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_loss_equals_naive_for_every_chunk_size(reduction, chunk_size):
    hidden, head, targets = _problem()
    want, _ = naive_linear_cross_entropy(hidden, head, targets)
    got, _ = chunked_linear_cross_entropy(
        hidden, head, targets, chunk_size=chunk_size, reduction=reduction
    )
    assert abs(float(got) - float(want)) <= 1e-5


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_a_single_chunk_larger_than_the_vocabulary_equals_naive(reduction):
    # chunk_size > |V| degenerates to one chunk of the whole vocabulary. The
    # fused kernel cannot take that in one launch, so it is checked separately.
    hidden, head, targets = _problem()
    want, _ = naive_linear_cross_entropy(hidden, head, targets)
    if reduction == "fused":
        with pytest.raises(ValueError):
            chunked_linear_cross_entropy(
                hidden, head, targets, chunk_size=BIG_CHUNK, reduction=reduction
            )
        return
    got, _ = chunked_linear_cross_entropy(
        hidden, head, targets, chunk_size=BIG_CHUNK, reduction=reduction
    )
    assert abs(float(got) - float(want)) <= 1e-5


@pytest.mark.parametrize("vocab", [8, 37, 256, 1024])
def test_loss_equals_naive_across_vocabulary_sizes(vocab):
    hidden, head, targets = _problem(seed=vocab, v=vocab)
    want, _ = naive_linear_cross_entropy(hidden, head, targets)
    got, _ = chunked_linear_cross_entropy(hidden, head, targets, chunk_size=64, reduction="mlx")
    assert abs(float(got) - float(want)) <= 1e-5


def test_one_chunk_over_the_whole_vocabulary_is_the_naive_path_exactly():
    hidden, head, targets = _problem(seed=1, v=64)
    want, want_per_token = naive_linear_cross_entropy(hidden, head, targets)
    got, got_per_token = chunked_linear_cross_entropy(
        hidden, head, targets, chunk_size=64, reduction="mlx"
    )
    # One chunk means the reduction is a single (max, sum-exp) pair, which is
    # exactly what the naive path computes.
    assert float(got) == float(want)
    np.testing.assert_array_equal(np.array(got_per_token), np.array(want_per_token))


def test_linear_cross_entropy_dispatches_on_chunk_size():
    hidden, head, targets = _problem(seed=2, v=32)
    naive_loss, naive_per_token = linear_cross_entropy(hidden, head, targets, chunk_size=None)
    ref_loss, ref_per_token = naive_linear_cross_entropy(hidden, head, targets)
    assert float(naive_loss) == float(ref_loss)
    np.testing.assert_array_equal(np.array(naive_per_token), np.array(ref_per_token))
    chunked_loss, _ = linear_cross_entropy(hidden, head, targets, chunk_size=5)
    assert abs(float(chunked_loss) - float(ref_loss)) <= 1e-5


# ---------------------------------------------------------------------- gradients


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("chunk_size", (1, 5, 64, 1000))
def test_hidden_gradient_equals_naive(reduction, chunk_size):
    """The test this repo exists for: d(loss)/d(hidden) must match the naive path."""
    hidden, head, targets = _problem(seed=3)

    def naive(h, c):
        return naive_linear_cross_entropy(h, c, targets)[0]

    def chunked(h, c):
        return chunked_linear_cross_entropy(
            h, c, targets, chunk_size=chunk_size, reduction=reduction
        )[0]

    _, ref_hidden, ref_head = _grad(naive, hidden, head)
    _, got_hidden, got_head = _grad(chunked, hidden, head)
    assert np.abs(got_hidden - ref_hidden).max() <= 1e-5
    assert np.abs(got_head - ref_head).max() <= 1e-5


def test_hidden_gradient_is_also_the_analytic_gradient():
    # An independent check of the same quantity, in float64, so that a shared
    # mistake in both MLX paths cannot pass.
    n, d, v = 4, 5, 30
    rng = np.random.default_rng(11)
    hidden = rng.normal(size=(n, d))
    head = rng.normal(size=(v, d))
    targets = rng.integers(0, v, size=n)

    z = hidden @ head.T
    m = z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z - m).sum(axis=1)) + m[:, 0]
    softmax = np.exp(z - lse[:, None])
    want_hidden = (softmax @ head) / n
    want_hidden[np.arange(n)] -= head[targets] / n

    def chunked(h, c):
        return chunked_linear_cross_entropy(
            h, c, mx.array(targets.astype(np.int32)), chunk_size=7
        )[0]

    _, got_hidden, _ = _grad(chunked, hidden.astype(np.float32), head.astype(np.float32))
    assert np.abs(got_hidden - want_hidden).max() <= 1e-5


def test_gradients_are_finite_and_nonzero():
    hidden, head, targets = _problem(seed=4)
    for reduction in REDUCTIONS:
        def chunked(h, c, reduction=reduction):
            return chunked_linear_cross_entropy(h, c, targets, chunk_size=16, reduction=reduction)[0]

        _, g_hidden, g_head = _grad(chunked, hidden, head)
        assert np.isfinite(g_hidden).all() and np.isfinite(g_head).all()
        assert np.abs(g_hidden).max() > 0 and np.abs(g_head).max() > 0


def test_ignored_tokens_get_no_gradient_and_do_not_scale_the_mean():
    n, d, v = 4, 3, 12
    rng = np.random.default_rng(5)
    hidden = mx.array(rng.normal(size=(n, d)).astype(np.float32))
    head = mx.array(rng.normal(size=(v, d)).astype(np.float32))
    targets = mx.array([1, -100, 3, -100], mx.int32)
    kept = mx.array([1, 3], mx.int32)

    def chunked(t):
        return lambda h, c: chunked_linear_cross_entropy(h, c, t, chunk_size=5)[0]

    def naive(t):
        return lambda h, c: naive_linear_cross_entropy(h, c, t)[0]

    masked_loss, masked_g_hidden, _ = _grad(chunked(targets), hidden, head)
    full_loss, full_g_hidden, _ = _grad(chunked(kept), hidden[[0, 2]], head)
    # The masked rows contribute nothing, so dropping them changes neither the
    # loss nor the gradient of the rows that remain.
    assert abs(masked_loss - full_loss) <= 1e-6
    assert np.abs(masked_g_hidden[[0, 2]] - full_g_hidden).max() <= 1e-6
    assert np.abs(masked_g_hidden[[1, 3]]).max() == 0.0
    _, ref_g_hidden, _ = _grad(naive(targets), hidden, head)
    assert np.abs(masked_g_hidden - ref_g_hidden).max() <= 1e-5


def test_all_targets_ignored_gives_zero_loss():
    hidden, head, _ = _problem(seed=6, n=4)
    targets = mx.full((4,), -100, mx.int32)
    loss, per_token = chunked_linear_cross_entropy(hidden, head, targets, chunk_size=9)
    assert float(loss) == 0.0
    assert not bool(mx.any(mx.isnan(per_token)))


# ------------------------------------------------------------------- stability


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_extreme_logits_are_stable(reduction):
    # Logits of +-1e3 push exp() far outside the float32 range, so this fails
    # with inf or nan unless the reduction subtracts a running maximum.
    n, d, v = 3, 4, 512
    rng = np.random.default_rng(7)
    hidden = mx.array(rng.normal(size=(n, d)).astype(np.float32))
    head = mx.array(rng.normal(size=(v, d)).astype(np.float32))
    targets = mx.array([0, 255, 511], mx.int32)
    scale = (mx.arange(v) % 2) * 2e3 - 1e3
    head = head + scale[:, None] * 2.0

    logits = np.array(hidden) @ np.array(head).T
    assert np.abs(logits).max() > 1e3
    z = logits.astype(np.float64)
    m = z.max(axis=1, keepdims=True)
    want = (np.log(np.exp(z - m).sum(axis=1)) + m[:, 0] - z[np.arange(n), np.array(targets)]).mean()

    loss, per_token = chunked_linear_cross_entropy(
        hidden, head, targets, chunk_size=64, reduction=reduction
    )
    assert float(mx.all(mx.isfinite(per_token)))
    assert abs(float(loss) - float(want)) <= 1e-3 * max(1.0, abs(float(want)))


def test_all_negative_infinite_logits_gives_zero_not_nan():
    hidden = mx.zeros((2, 3), mx.float32)
    head = mx.full((8, 3), -mx.inf, mx.float32)
    targets = mx.array([0, 7], mx.int32)
    _, per_token = chunked_linear_cross_entropy(hidden, head, targets, chunk_size=4)
    assert not bool(mx.any(mx.isnan(per_token)))


def test_reductions_agree_with_each_other():
    hidden, head, targets = _problem(seed=8, v=100)

    def loss(reduction, chunk_size):
        return float(
            chunked_linear_cross_entropy(
                hidden, head, targets, chunk_size=chunk_size, reduction=reduction
            )[0]
        )

    fused, mlx_, inline = loss("fused", 32), loss("mlx", 32), loss("inline", 32)
    assert abs(fused - mlx_) <= 1e-6
    assert abs(inline - mlx_) <= 1e-6


# ----------------------------------------------------------- gradient filtering


def _peaked_problem(targets=(1, 9, 20, 41, 63, 55), n=None, d=8, v=64, seed=9):
    """A problem with a peaked softmax, so most vocabulary blocks are negligible.

    The classifier is built so that the ground-truth logit of every row is about
    20 above the rest, which is the regime the paper's gradient filtering relies
    on: one dominant softmax weight and a long tail that rounds away.
    """
    n = len(targets) if n is None else n
    rng = np.random.default_rng(seed)
    hidden = mx.array(rng.normal(size=(n, d)).astype(np.float32) * 0.1)
    head = mx.array(rng.normal(size=(v, d)).astype(np.float32) * 0.1)
    targets = mx.array(list(targets)[:n], mx.int32)
    one_hot = (mx.arange(v)[None, :] == targets[:, None]).astype(mx.float32)
    head = head + one_hot.T @ (hidden * 200.0)
    return hidden, head, targets


def test_threshold_zero_is_exactly_no_filtering():
    hidden, head, targets = _peaked_problem()

    def grad(threshold):
        return _grad(
            lambda h, c: chunked_linear_cross_entropy(
                h, c, targets, chunk_size=16, grad_threshold=threshold
            )[0],
            hidden,
            head,
        )

    base_loss, base_hidden, base_head = grad(0.0)
    for threshold in (0.0, -1.0):
        loss, g_hidden, g_head = grad(threshold)
        assert loss == base_loss
        np.testing.assert_array_equal(g_hidden, base_hidden)
        np.testing.assert_array_equal(g_head, base_head)
    # Nothing can be skipped when the filter is off.
    assert last_filter_report()["skipped_blocks"] == 0.0


def test_bf16_threshold_skips_blocks_and_stays_close():
    hidden, head, targets = _peaked_problem()
    ref = _grad(
        lambda h, c: chunked_linear_cross_entropy(h, c, targets, chunk_size=8)[0], hidden, head
    )
    got = _grad(
        lambda h, c: chunked_linear_cross_entropy(
            h, c, targets, chunk_size=8, grad_threshold=BFLOAT16_EPSILON
        )[0],
        hidden,
        head,
    )
    report = last_filter_report()
    assert report["skipped_blocks"] == 2
    assert report["skipped_block_fraction"] == 0.25
    # Loss is untouched by filtering; the gradient moves only by the truncated
    # mass of the softmax.
    assert got[0] == ref[0]
    assert np.abs(got[1] - ref[1]).max() <= 1e-3
    assert np.abs(got[2] - ref[2]).max() <= 1e-3


def test_skip_decisions_match_an_independent_float64_rule():
    """Every skip must be justified by the softmax of the ground-truth model.

    This recomputes the paper's ``all(S_nm < eps)`` test in float64 from the
    logits and checks the count of skipped blocks, so the test does not just
    re-state the implementation.
    """
    hidden, head, targets = _peaked_problem()
    chunk_size, eps = 8, BFLOAT16_EPSILON
    vocab = int(head.shape[0])
    chunks = list(range(0, vocab, chunk_size))

    logits = np.array(hidden).astype(np.float64) @ np.array(head).astype(np.float64).T
    m = logits.max(axis=1, keepdims=True)
    softmax = np.exp(logits - m)
    softmax /= softmax.sum(axis=1, keepdims=True)
    negligible = softmax < eps
    expected_skipped = sum(
        1 for c in chunks if negligible[:, c : c + chunk_size].all()
    )

    _, g_hidden, g_head = _grad(
        lambda h, c: chunked_linear_cross_entropy(
            h, c, targets, chunk_size=chunk_size, grad_threshold=eps
        )[0],
        hidden,
        head,
    )
    report = last_filter_report()
    assert report["blocks"] == len(chunks)
    assert report["skipped_blocks"] == expected_skipped
    assert expected_skipped > 0
    # The gathered ground-truth term of Equation 4 is never filtered, so the
    # hidden gradient keeps a nonzero contribution on every row.
    assert np.abs(g_hidden).max() > 0.0
    assert np.isfinite(g_hidden).all() and np.isfinite(g_head).all()


def test_a_threshold_above_the_whole_softmax_drops_every_chunk():
    """With eps > 1 no softmax weight can survive, so only the gathered term is left."""
    hidden, head, targets = _peaked_problem()
    ref_hidden_gather = -(
        np.array(head)[np.array(targets)] / len(np.array(targets))
    )
    _, g_hidden, g_head = _grad(
        lambda h, c: chunked_linear_cross_entropy(
            h, c, targets, chunk_size=8, grad_threshold=2.0
        )[0],
        hidden,
        head,
    )
    assert last_filter_report()["skipped_blocks"] == last_filter_report()["blocks"]
    assert np.abs(g_head).max() == 0.0
    assert np.abs(g_hidden - ref_hidden_gather).max() <= 1e-6


# ---------------------------------------------------------------------- shapes


def test_shape_and_value_errors():
    hidden, head, targets = _problem(seed=10, n=4, d=8, v=16)
    with pytest.raises(ValueError):
        chunked_linear_cross_entropy(hidden, head[:, :4], targets)
    with pytest.raises(ValueError):
        chunked_linear_cross_entropy(hidden, head, targets[:2])
    with pytest.raises(ValueError):
        chunked_linear_cross_entropy(hidden, head, targets, chunk_size=0)
    with pytest.raises(ValueError):
        chunked_linear_cross_entropy(hidden, head, targets, reduction="nope")
    with pytest.raises(ValueError):
        chunked_linear_cross_entropy(hidden, head, targets, chunk_size=2048, reduction="fused")
    with pytest.raises(ValueError):
        chunked_linear_cross_entropy(hidden, head, targets, chunk_size=-4)
    with pytest.raises(ValueError):
        naive_linear_cross_entropy(hidden, head.reshape(2, 2, 4), targets)
