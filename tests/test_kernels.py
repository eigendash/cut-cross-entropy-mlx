import numpy as np
import pytest

import mlx.core as mx

from cce.kernels import LSE_THREADGROUP, col_sum, row_stats


def _reference_stats(x: np.ndarray):
    z = x.astype(np.float64)
    m = z.max(axis=1)
    s = np.exp(z - m[:, None]).sum(axis=1)
    return m, s


@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (4, 256), (2, 1000), (5, 3), (17, 64)])
def test_row_stats_matches_a_float64_reference(shape):
    rng = np.random.default_rng(0)
    x = (rng.normal(size=shape) * 5).astype(np.float32)
    m, s = row_stats(mx.array(x))
    want_m, want_s = _reference_stats(x)
    np.testing.assert_allclose(np.array(m), want_m, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(s), want_s, rtol=1e-5)


def test_row_stats_is_stable_on_extreme_logits():
    # +-1e3 is far outside the range where exp() is finite, so anything that
    # exponentiates an unshifted logit produces inf or nan here.
    x = np.concatenate(
        [np.full((2, 500), -1e3), np.full((2, 500), 1e3), np.full((2, 24), -1e3)], axis=1
    ).astype(np.float32)
    m, s = row_stats(mx.array(x))
    np.testing.assert_allclose(np.array(m), [1e3, 1e3], rtol=1e-6)
    np.testing.assert_allclose(np.array(s), [500.0, 500.0], rtol=1e-5)
    lse = np.array(m) + np.log(np.array(s))
    np.testing.assert_allclose(lse, [1e3 + np.log(500.0)] * 2, rtol=1e-6)


def test_row_stats_preserves_the_all_negative_infinity_row():
    x = mx.array([[-mx.inf, -mx.inf], [0.0, 1.0]], mx.float32)
    m, s = row_stats(x)
    assert float(m[0]) == -float("inf")
    assert float(s[0]) == 0.0
    assert not bool(mx.any(mx.isnan(m)))
    assert abs(float(m[1]) - 1.0) < 1e-6


def test_row_stats_reports_a_bad_block_size():
    with pytest.raises(ValueError):
        row_stats(mx.zeros((2, LSE_THREADGROUP + 1), mx.float32))
    with pytest.raises(ValueError):
        row_stats(mx.zeros((2, 0), mx.float32))
    with pytest.raises(ValueError):
        row_stats(mx.zeros((2, 3, 4), mx.float32))


@pytest.mark.parametrize("shape", [(1, 1), (7, 3), (256, 4), (1000, 2), (33, 37)])
def test_col_sum_matches_mlx(shape):
    rng = np.random.default_rng(1)
    x = rng.uniform(size=shape).astype(np.float32)
    got = col_sum(mx.array(x))
    want = np.array(mx.sum(mx.array(x), axis=0))
    np.testing.assert_allclose(np.array(got), want, rtol=1e-5, atol=1e-4)


def test_col_sum_handles_an_empty_batch():
    out = col_sum(mx.zeros((0, 5), mx.float32))
    np.testing.assert_array_equal(np.array(out), np.zeros(5))
    with pytest.raises(ValueError):
        col_sum(mx.zeros((2, LSE_THREADGROUP + 4), mx.float32))
    with pytest.raises(ValueError):
        col_sum(mx.zeros((2, 0), mx.float32))
    with pytest.raises(ValueError):
        col_sum(mx.zeros((2,), mx.float32))


def test_row_stats_is_row_contiguous_safe():
    # A non-contiguous view (the logits chunk of a wider matrix) must still be
    # reduced correctly, which is what the chunked path actually feeds it.
    rng = np.random.default_rng(2)
    wide = rng.normal(size=(4, 512)).astype(np.float32)
    view = mx.array(wide)[:, 100:356]
    m, s = row_stats(view)
    want_m, want_s = _reference_stats(wide[:, 100:356])
    np.testing.assert_allclose(np.array(m), want_m, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(s), want_s, rtol=1e-5)
