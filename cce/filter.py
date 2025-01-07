"""Sparse-gradient skipping ("gradient filtering", paper Section 4.2).

The softmax weights of a language model are extremely sparse: the paper
observes that at most ``1/eps`` entries per token can exceed a bfloat16
threshold ``eps = 2**-12``, that fewer than 0.02% of the entries of real
frontier-model softmax matrices are non-zero, and that the sparsity grows with
the vocabulary. So a whole vocabulary chunk can usually be skipped in the
backward pass when every one of its softmax weights is negligible.

The skip here is *block level and conservative*: a chunk is skipped only when
every softmax weight in it is strictly below the threshold, which is the same
rule as the paper's ``if all(S_nm < eps) then skip``. With ``threshold = 0`` the
test can never fire (a nonzero softmax entry is never ``< 0``), so the default
path is the exact one.
"""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx

# Smallest bfloat16 value that is not truncated by rounding, from the paper.
BFLOAT16_EPSILON = 2.0 ** -12


def softmax_is_negligible(weights: mx.array, threshold: float) -> mx.array:
    """Boolean mask that is true where ``weights < threshold``.

    A non-positive threshold disables the filter, which keeps the default path
    identical to running without filtering.
    """
    if threshold <= 0.0:
        return mx.zeros(weights.shape, mx.bool_)
    return weights < threshold


def gradient_filter_report(
    skipped: mx.array,
    n_tokens: int,
) -> dict[str, float]:
    """Block-level skip statistics for one backward pass.

    ``skipped`` is a ``(num_chunks,)`` boolean array that is true for a
    vocabulary chunk that no token needed, so the chunk was dropped entirely
    from the backward pass.
    """
    n_blocks = int(skipped.shape[0])
    n_skipped = int(mx.sum(skipped.astype(mx.int32)))
    return {
        "blocks": float(n_blocks),
        "skipped_blocks": float(n_skipped),
        "skipped_block_fraction": float(n_skipped / n_blocks) if n_blocks else 0.0,
        "tokens": float(n_tokens),
    }
