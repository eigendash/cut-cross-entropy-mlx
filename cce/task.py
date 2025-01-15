"""A synthetic next-token task with a deliberately large vocabulary.

The point of the task is to give the cross-entropy head something to learn while
the vocabulary - and therefore the logits matrix - is large enough to dominate
memory. Each "key" token is followed by 63 tokens drawn i.i.d. from that key's
own distribution over a small support of the vocabulary, so the conditional
next-token entropy is about ``log(64)`` nats and the task is learnable by
looking at the previous token. A model that ignored the previous token would sit
at ``log(|V|)``, which is 7 nats worse for a 64k vocabulary.
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

SUPPORT = 64


class Batch(NamedTuple):
    tokens: mx.array
    targets: mx.array
    mask: mx.array


class BigramTask:
    """A bigram language model with a sparse, key-conditioned next-token map.

    ``n_keys`` keys occupy the first ``n_keys`` vocabulary slots. Each key ``k``
    has a support of ``SUPPORT`` tokens drawn from the rest of the vocabulary,
    with a light Dirichlet concentration, and the segment that follows a key is
    sampled i.i.d. from that key's distribution.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        n_keys: int = 32,
        support: int = SUPPORT,
        segment: int = 64,
        concentration: float = 0.4,
        seed: int = 0,
    ):
        if vocab_size < n_keys + support:
            raise ValueError("vocabulary is too small for the requested keys and support")
        if segment < 2:
            raise ValueError("a segment needs at least a key and one continuation")
        self.vocab_size = vocab_size
        self.n_keys = n_keys
        self.support = support
        self.segment = segment
        # MLX has no gamma sampler, so the per-key weights are exponentials with
        # a random rate (the concentration parameter) renormalised across the
        # support, which is the same shape of distribution.
        mx.random.seed(seed)
        rates = concentration * (0.25 + 2.0 * mx.random.uniform(shape=(n_keys, support)))
        weights = mx.exp(-rates)
        # Support tokens are drawn without replacement from everything after the
        # keys, and are not shared between keys, so the previous token really
        # does determine the conditional distribution.
        pool = mx.random.permutation(mx.arange(n_keys, vocab_size, dtype=mx.int32))
        chosen = pool[: n_keys * support].reshape(n_keys, support)
        self.key_tokens = mx.arange(n_keys, dtype=mx.int32)
        self.support_tokens = chosen
        self.weights = weights / mx.sum(weights, axis=-1, keepdims=True)

    def sample(self, batch_size: int, *, seed: int = 0) -> Batch:
        mx.random.seed(seed)
        keys = mx.random.randint(0, self.n_keys, (batch_size,))
        # (batch, segment - 1) support positions, one multinomial draw per key.
        draws = mx.random.categorical(
            mx.log(self.weights)[keys], axis=-1, num_samples=self.segment - 1
        )
        # (batch, support) support tokens per key, indexed by (batch, num_samples)
        continuation = self.support_tokens[keys][mx.arange(keys.shape[0])[:, None], draws]
        tokens = mx.concatenate([keys[:, None], continuation], axis=1)
        # Predict every position from the previous one, except the first, which
        # has no context in this batch.
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        mask = mx.ones(targets.shape, mx.bool_)
        return Batch(inputs, targets, mask)

    def ideal_loss(self) -> float:
        """Entropy of the conditional distribution a perfect model would reach."""
        return float(-mx.mean(mx.sum(self.weights * mx.log(self.weights), axis=-1)))

    def marginal_weights(self) -> mx.array:
        """Next-token distribution of a model that ignores the previous token.

        Each key is followed by the same number of continuations, so the
        marginal is the mean over keys of the key's own distribution.
        """
        accumulated = mx.zeros((self.vocab_size,), mx.float32)
        tokens = mx.arange(self.vocab_size)
        for k in range(self.n_keys):
            basis = (tokens[None, :] == self.support_tokens[k][:, None]).astype(mx.float32)
            accumulated = accumulated + basis.T @ self.weights[k] / self.n_keys
        return accumulated / mx.sum(accumulated)

    def uniform_loss(self) -> float:
        """Loss of a model that ignores the previous token entirely."""
        p = self.marginal_weights()
        # p has exact zeros for the 60k+ tokens outside every key's support.
        return float(-mx.sum(p * mx.log(p + 1e-30)))


def masked_mean_loss(per_token: mx.array, mask: mx.array) -> mx.array:
    """Mean of per-token losses over the positions the batch actually scores."""
    weights = mask.astype(mx.float32)
    n = mx.maximum(mx.sum(weights), 1.0)
    return mx.sum(per_token * weights) / n
