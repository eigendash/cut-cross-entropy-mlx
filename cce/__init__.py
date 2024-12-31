"""Cut Cross-Entropy in MLX.

A small independent implementation of *Cut Your Losses in Large-Vocabulary
Language Models* (arXiv:2411.09009): cross-entropy computed by streaming over
the vocabulary so that the ``batch x vocab`` logits matrix is never
materialised, plus a fused Metal kernel for the chunk reduction and an
optional sparse-gradient skip.

Public API
----------
``naive_cross_entropy``
    Reference path that materialises the full logits matrix.
``chunked_cross_entropy``
    Streaming path: reads the logits one vocabulary chunk at a time.
``OnlineLSE``
    Running-max / running-sum accumulator used by the streaming forward pass.
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

__all__ = [
    "IGNORE_INDEX",
    "OnlineLSE",
    "naive_cross_entropy",
    "chunked_cross_entropy",
    "cross_entropy_from_lse",
    "logsumexp_from_lse",
    "online_update",
]
