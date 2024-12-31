import math

import mlx.core as mx
import numpy as np

from cce.ce import (
    IGNORE_INDEX,
    OnlineLSE,
    chunked_cross_entropy,
    cross_entropy_from_lse,
    naive_cross_entropy,
    online_update,
)


def test_hand_computed_case():
    # Two tokens, three vocabulary entries, uniform logits. With targets 0 and 2
    # the loss is log(3) - (0 + 0) / 2 on the first row and log(3) - 0 on the
    # second, so the mean is exactly log(3).
    logits = mx.zeros((2, 3), mx.float32)
    targets = mx.array([0, 2], mx.int32)
    loss, per_token = naive_cross_entropy(logits, targets)
    assert abs(float(loss) - math.log(3.0)) < 1e-6
    np.testing.assert_allclose(np.array(per_token), [math.log(3.0)] * 2, atol=1e-6)


def test_matches_float64_numpy_reference():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 37)).astype(np.float32)
    targets = rng.integers(0, 37, size=5).astype(np.int32)
    # A hand-rolled float64 reference: subtract the row max, log the sum of exps.
    z = logits.astype(np.float64)
    m = z.max(axis=1, keepdims=True)
    want = np.log(np.exp(z - m).sum(axis=1)) + m[:, 0] - z[np.arange(5), targets]
    loss, per_token = naive_cross_entropy(mx.array(logits), mx.array(targets))
    np.testing.assert_allclose(np.array(per_token), want, rtol=1e-6, atol=1e-6)
    assert abs(float(loss) - want.mean()) < 1e-6


def test_ignored_tokens_are_excluded_from_the_mean():
    logits = mx.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], mx.float32)
    targets = mx.array([1, IGNORE_INDEX], mx.int32)
    loss, per_token = naive_cross_entropy(logits, targets)
    # Only the first row contributes, so the mean is over one token, not two.
    assert abs(float(loss) - math.log(3.0)) < 1e-6
    assert float(per_token[1]) == 0.0


def test_all_ignored_gives_zero_loss_and_no_nan():
    logits = mx.zeros((3, 8), mx.float32)
    targets = mx.full((3,), IGNORE_INDEX, mx.int32)
    loss, per_token = naive_cross_entropy(logits, targets)
    assert float(loss) == 0.0
    assert not bool(mx.any(mx.isnan(loss)))
    assert float(mx.sum(per_token)) == 0.0


def test_targets_are_gathered_from_the_right_column():
    logits = mx.array([[1.0, 2.0, 3.0, 4.0]], mx.float32)
    for target in range(4):
        loss, _ = naive_cross_entropy(logits, mx.array([target], mx.int32))
        want = math.log(sum(math.exp(v) for v in [1.0, 2.0, 3.0, 4.0])) - float(target + 1)
        assert abs(float(loss) - want) < 1e-6


def test_online_update_is_a_merge_of_summaries():
    # Merging summaries must equal summarising the concatenation, and must not
    # care about the order in which the chunks arrive.
    a = mx.array([[0.0, 1.0, -5.0]], mx.float32)
    b = mx.array([[3.0, -2.0, 0.5]], mx.float32)

    def summary(x):
        m = mx.max(x, axis=1)
        return OnlineLSE(max=m, sum_exp=mx.sum(mx.exp(x - m[:, None]), axis=1))

    from cce.ce import logsumexp_from_lse

    whole = logsumexp_from_lse(summary(mx.concatenate([a, b], axis=1)))
    sa, sb = summary(a), summary(b)
    forward = logsumexp_from_lse(online_update(sa, sb.max, sb.sum_exp))
    backward = logsumexp_from_lse(online_update(sb, sa.max, sa.sum_exp))
    np.testing.assert_allclose(np.array(forward), np.array(whole), rtol=1e-6)
    np.testing.assert_allclose(np.array(backward), np.array(whole), rtol=1e-6)


def test_chunked_matches_naive_for_every_chunk_size():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(6, 50)).astype(np.float32) * 4
    targets = rng.integers(0, 50, size=6).astype(np.int32)
    want, _ = naive_cross_entropy(mx.array(logits), mx.array(targets))
    for chunk_size in (1, 2, 7, 13, 49, 50, 64, 1000):
        def chunk(start, stop, logits=logits):
            return mx.array(logits[:, start:stop])

        got, _ = chunked_cross_entropy(chunk, mx.array(targets), chunk_size=chunk_size)
        assert abs(float(got) - float(want)) < 1e-6, chunk_size


def test_one_chunk_covering_the_vocabulary_is_the_naive_path():
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(4, 16)).astype(np.float32)
    targets = rng.integers(0, 16, size=4).astype(np.int32)
    naive, naive_per_token = naive_cross_entropy(mx.array(logits), mx.array(targets))
    chunked, chunked_per_token = chunked_cross_entropy(
        lambda start, stop: mx.array(logits[:, start:stop]), mx.array(targets), chunk_size=16
    )
    assert float(naive) == float(chunked)
    np.testing.assert_array_equal(np.array(naive_per_token), np.array(chunked_per_token))


def test_cross_entropy_from_lse_rejects_bad_shapes():
    import pytest

    with pytest.raises(ValueError):
        naive_cross_entropy(mx.zeros((2, 3, 4), mx.float32), mx.array([0, 1], mx.int32))
    with pytest.raises(ValueError):
        naive_cross_entropy(mx.zeros((2, 3), mx.float32), mx.array([0], mx.int32))
    with pytest.raises(ValueError):
        chunked_cross_entropy(lambda s, e: mx.zeros((2, 0), mx.float32), mx.array([0, 1], mx.int32), chunk_size=4)
    with pytest.raises(ValueError):
        chunked_cross_entropy(lambda s, e: mx.zeros((2, 4), mx.float32), mx.array([0, 1], mx.int32), chunk_size=0)


def test_cross_entropy_from_lse_masks_and_normalises():
    lse = mx.array([2.0, 3.0], mx.float32)
    target_logits = mx.array([1.0, 0.5], mx.float32)
    ignore = mx.array([False, True])
    loss, per_token = cross_entropy_from_lse(lse, target_logits, ignore)
    assert abs(float(loss) - 1.0) < 1e-6
    np.testing.assert_allclose(np.array(per_token), [1.0, 0.0], atol=0.0)
