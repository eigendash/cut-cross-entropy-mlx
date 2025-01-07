"""Cut Cross-Entropy in MLX.

A small independent implementation of *Cut Your Losses in Large-Vocabulary
Language Models* (arXiv:2411.09009). Standard cross-entropy materialises the
full ``batch x vocab`` logits matrix; this package computes the same loss (and
its gradients) by streaming over the vocabulary in chunks and accumulating the
log-sum-exp online, so the logits matrix never exists.

Public API
----------
``naive_cross_entropy`` / ``naive_linear_cross_entropy``
    Reference paths that materialise the full logits matrix.
``chunked_cross_entropy``
    Streaming loss given a callable that yields one vocabulary chunk at a time.
``chunked_linear_cross_entropy`` / ``linear_cross_entropy``
    The chunked linear + cross-entropy operator with an explicit ``vjp``.
``OnlineLSE``
    Running-max / running-sum accumulator behind the streaming reduction.
``row_stats`` / ``col_sum``
    The fused Metal kernels used by the chunked reduction and the backward pass.
``softmax_is_negligible`` / ``gradient_filter_report``
    Sparse-gradient skipping ("gradient filtering", paper Section 4.2) and its
    bookkeeping.
"""

from .ce import (
    IGNORE_INDEX,
    OnlineLSE,
    chunked_cross_entropy,
    cross_entropy_from_lse,
    logsumexp_from_lse,
    naive_cross_entropy,
    online_update,
)
from .filter import BFLOAT16_EPSILON, gradient_filter_report, softmax_is_negligible
from .kernels import col_sum, row_stats
from .linear_ce import (
    MAX_FUSED_BLOCK,
    REDUCTIONS,
    chunked_linear_cross_entropy,
    last_filter_report,
    linear_cross_entropy,
    naive_linear_cross_entropy,
)

__all__ = [
    "IGNORE_INDEX",
    "MAX_FUSED_BLOCK",
    "REDUCTIONS",
    "BFLOAT16_EPSILON",
    "OnlineLSE",
    "online_update",
    "logsumexp_from_lse",
    "cross_entropy_from_lse",
    "naive_cross_entropy",
    "chunked_cross_entropy",
    "naive_linear_cross_entropy",
    "chunked_linear_cross_entropy",
    "linear_cross_entropy",
    "row_stats",
    "col_sum",
    "softmax_is_negligible",
    "gradient_filter_report",
    "last_filter_report",
]
