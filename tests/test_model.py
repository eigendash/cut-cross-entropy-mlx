import math

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from cce.linear_ce import chunked_linear_cross_entropy, naive_linear_cross_entropy
from cce.model import Config, TinyLM
from cce.task import BigramTask, masked_mean_loss


def _small_cfg(vocab=512, seq_len=32, d_model=32, d_ff=64):
    return Config(
        vocab_size=vocab, d_model=d_model, n_layers=2, n_heads=4, d_ff=d_ff, max_seq_len=seq_len
    )


def _backbone_params(model) -> int:
    """Parameters that are not the token embedding or the classifier head."""
    params = dict(nn.utils.tree_flatten(model.parameters()))
    return sum(
        int(v.size)
        for k, v in params.items()
        if not k.startswith(("embed", "head"))
    )


def test_model_forward_shapes_and_parameter_count():
    cfg = _small_cfg()
    model = TinyLM(cfg)
    tokens = mx.zeros((2, 16), mx.int32)
    assert model.hidden(tokens).shape == (2, 16, cfg.d_model)
    assert model(tokens).shape == (2, 16, cfg.vocab_size)
    assert model.n_params == _backbone_params(model) + 2 * cfg.vocab_size * cfg.d_model


def test_backbone_is_small_relative_to_the_classifier_head():
    # This is the whole premise of the experiment: at the scale used here the
    # classifier head is the memory, not the transformer.
    cfg = Config(vocab_size=65536, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=128)
    model = TinyLM(cfg)
    backbone = _backbone_params(model)
    head = cfg.vocab_size * cfg.d_model
    assert backbone < 80_000
    assert model.n_params == backbone + 2 * head
    assert head > 50 * backbone


def test_model_starts_uniform_so_the_first_loss_is_log_vocab():
    cfg = _small_cfg(vocab=256)
    model = TinyLM(cfg)
    task = BigramTask(cfg.vocab_size, n_keys=8, support=16, segment=17, seed=0)
    batch = task.sample(3, seed=1)
    logits = model(batch.tokens)
    per_token = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(
        logits, batch.targets[:, :, None], axis=-1
    )[:, :, 0]
    loss = masked_mean_loss(per_token, batch.mask)
    assert abs(float(loss) - math.log(cfg.vocab_size)) < 1e-4


def test_model_is_causal():
    # Changing a future token must not change an earlier position's hidden state.
    cfg = _small_cfg(vocab=64, seq_len=16)
    model = TinyLM(cfg)
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], mx.int32)
    other = mx.array([[1, 2, 3, 9, 9, 9, 9, 9]], mx.int32)
    a = np.array(model.hidden(tokens))[0, :3]
    b = np.array(model.hidden(other))[0, :3]
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_model_rejects_too_long_sequences():
    cfg = _small_cfg(seq_len=8)
    model = TinyLM(cfg)
    with pytest.raises(ValueError):
        model.hidden(mx.zeros((1, 9), mx.int32))


def test_chunked_loss_matches_naive_on_the_toy_model():
    cfg = _small_cfg(vocab=300)
    model = TinyLM(cfg)
    task = BigramTask(cfg.vocab_size, n_keys=8, support=16, segment=9, seed=0)
    batch = task.sample(2, seed=2)
    hidden = model.hidden(batch.tokens).reshape(-1, cfg.d_model)
    targets = batch.targets.reshape(-1)
    naive, _ = naive_linear_cross_entropy(hidden, model.head.weight, targets)
    for chunk_size in (16, 64, 128):
        got, _ = chunked_linear_cross_entropy(
            hidden, model.head.weight, targets, chunk_size=chunk_size
        )
        assert abs(float(got) - float(naive)) <= 1e-5


def test_model_gradients_flow_to_every_parameter():
    cfg = _small_cfg(vocab=128, d_model=32, d_ff=64)
    model = TinyLM(cfg)
    task = BigramTask(cfg.vocab_size, n_keys=4, support=8, segment=5, seed=0)
    batch = task.sample(2, seed=3)

    def loss_fn(model, tokens, targets, mask):
        hidden = model.hidden(tokens) @ model.head.weight.T
        per_token = mx.logsumexp(hidden, axis=-1) - mx.take_along_axis(
            hidden, targets[:, :, None], axis=-1
        )[:, :, 0]
        return masked_mean_loss(per_token, mask)

    loss, grads = nn.value_and_grad(model, loss_fn)(model, batch.tokens, batch.targets, batch.mask)
    assert math.isfinite(float(loss))
    flat = nn.utils.tree_flatten(grads)
    assert flat, "no gradients were produced"
    for _, grad in flat:
        assert mx.all(mx.isfinite(grad)).item()
    assert any(float(mx.max(mx.abs(g))) > 0 for _, g in flat)


def test_masked_mean_loss_ignores_masked_positions():
    per_token = mx.array([[1.0, 2.0, 100.0]], mx.float32)
    mask = mx.array([[True, True, False]])
    assert abs(float(masked_mean_loss(per_token, mask)) - 1.5) < 1e-6
    all_off = mx.array([[False, False, False]])
    assert float(masked_mean_loss(per_token, all_off)) == 0.0


def test_task_is_deterministic_and_uses_the_previous_token():
    task = BigramTask(2048, n_keys=16, support=32, segment=16, seed=0)
    a = task.sample(4, seed=7)
    b = task.sample(4, seed=7)
    np.testing.assert_array_equal(np.array(a.tokens), np.array(b.tokens))
    c = task.sample(4, seed=8)
    assert not np.array_equal(np.array(a.tokens), np.array(c.tokens))

    tokens = np.array(a.tokens)
    keys = tokens[:, 0]
    assert np.all(keys < task.n_keys)
    supports = np.array(task.support_tokens)
    for row in range(tokens.shape[0]):
        for token in tokens[row, 1:]:
            assert token in set(supports[keys[row]].tolist())


def test_task_reports_a_realisable_floor_and_a_worse_ignoring_context_loss():
    task = BigramTask(4096, n_keys=32, support=64, segment=32, seed=0)
    ideal = task.ideal_loss()
    ignoring = task.uniform_loss()
    assert abs(ideal - math.log(64)) < 0.05
    assert ignoring > ideal
    assert ignoring < math.log(4096)
    p = np.array(task.marginal_weights())
    assert abs(p.sum() - 1.0) < 1e-5
    assert (p >= 0).all()


def test_task_rejects_an_impossible_configuration():
    with pytest.raises(ValueError):
        BigramTask(vocab_size=32, n_keys=32, support=64)
    with pytest.raises(ValueError):
        BigramTask(vocab_size=1024, segment=1)
