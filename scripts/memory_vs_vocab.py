"""Cut Cross-Entropy vs the naive logits-materialising path, at toy scale.

Three measurements, all on this machine with MLX:

1. **Equality.** Loss and gradients of the chunked path against the naive path,
   for several chunk sizes and reduction implementations.
2. **Peak memory.** ``mx.get_peak_memory()`` for a forward+backward pass as the
   vocabulary grows from 8k to 131k. This is the falsifiable claim: the naive
   path should grow linearly in ``|V|``, the chunked path sub-linearly.
3. **Wall clock**, to show what the memory costs in time here.

A short training run at a small vocabulary is included as a liveness check: both
paths must actually train, and to the same loss.

Usage::

    python scripts/memory_vs_vocab.py [--quick] [--out results]

Writes ``results/memory_vs_vocab.json`` and ``results/run_log.txt``.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cce.filter import BFLOAT16_EPSILON  # noqa: E402
from cce.linear_ce import (  # noqa: E402
    REDUCTIONS,
    chunked_linear_cross_entropy,
    last_filter_report,
    naive_linear_cross_entropy,
)
from cce.model import Config, TinyLM  # noqa: E402
from cce.task import BigramTask, masked_mean_loss  # noqa: E402

D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
D_FF = 128
SEQ_LEN = 128
BATCH = 4
TOKENS = 256


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.lines: list[str] = []

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        self.lines.append(message)

    def write(self) -> None:
        self.path.write_text("\n".join(self.lines) + "\n")


def make_model(vocab: int, hidden_size: int, seed: int = 0) -> TinyLM:
    mx.random.seed(seed)
    cfg = Config(
        vocab_size=vocab,
        d_model=hidden_size,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        max_seq_len=SEQ_LEN,
    )
    model = TinyLM(cfg)
    # The stock head is all zeros (that is what makes the first loss exactly
    # log |V|). Give it a small random draw so the softmax has structure and the
    # gradient filter has something to filter.
    model.head.weight = mx.random.normal(shape=(vocab, hidden_size)) * 0.02
    mx.eval(model.parameters())
    return model


def make_batch(hidden_size: int, vocab: int, tokens: int, seed: int = 0):
    """A fixed ``(tokens, D)`` hidden state and integer targets.

    The hidden state is detached on purpose: this measures the cost of the
    cross-entropy head, not of the backbone that produced it.
    """
    mx.random.seed(seed + 1000)
    hidden = mx.stop_gradient(mx.random.normal(shape=(tokens, hidden_size)) * 0.1)
    targets = mx.random.randint(0, vocab, (tokens,))
    mx.eval(hidden, targets)
    return hidden, targets


def measure(fn, hidden: mx.array, head: mx.array, warmup: int = 3) -> dict[str, float]:
    """Peak and active memory plus wall clock for one forward+backward pass.

    ``mx.get_peak_memory()`` is a high-water mark since the last reset, so it is
    reset immediately before the measured call, after the compilation warm-up.
    The warm-up matters: a first call that includes Metal kernel compilation
    shows a peak tens of megabytes above the steady state.
    """
    for _ in range(warmup):
        mx.eval(mx.value_and_grad(fn, argnums=(0, 1))(hidden, head))
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    active_before = mx.get_active_memory()
    start = time.perf_counter()
    mx.eval(mx.value_and_grad(fn, argnums=(0, 1))(hidden, head))
    elapsed = time.perf_counter() - start
    return {
        "peak_mb": mx.get_peak_memory() / 1e6,
        "active_mb": mx.get_active_memory() / 1e6,
        "active_before_mb": active_before / 1e6,
        "ms": elapsed * 1e3,
    }


def equality_section(log: Logger, vocabularies, chunk_sizes) -> list[dict]:
    """Loss and gradient equality between the two paths."""
    rows: list[dict] = []
    for vocab in vocabularies:
        model = make_model(vocab, D_MODEL, seed=vocab)
        head = model.head.weight
        hidden, targets = make_batch(D_MODEL, vocab, TOKENS, seed=vocab)

        def naive_fn(h, c):
            return naive_linear_cross_entropy(h, c, targets)[0]

        ref_loss, (ref_g_hidden, ref_g_head) = mx.value_and_grad(naive_fn, argnums=(0, 1))(hidden, head)
        mx.eval(ref_loss, ref_g_hidden, ref_g_head)
        log(f"  vocab={vocab:6d} naive loss={float(ref_loss):.6f} head-grad norm={float(mx.sqrt(mx.sum(ref_g_head ** 2))):.4f}")
        for reduction in REDUCTIONS:
            for chunk_size in chunk_sizes:
                if reduction == "fused" and chunk_size > 1024:
                    continue

                def chunked_fn(h, c, chunk_size=chunk_size, reduction=reduction):
                    return chunked_linear_cross_entropy(
                        h, c, targets, chunk_size=chunk_size, reduction=reduction
                    )[0]

                loss, (g_hidden, g_head) = mx.value_and_grad(chunked_fn, argnums=(0, 1))(hidden, head)
                mx.eval(loss, g_hidden, g_head)
                rows.append(
                    {
                        "vocab": vocab,
                        "reduction": reduction,
                        "chunk_size": chunk_size,
                        "loss": float(loss),
                        "loss_abs_diff": abs(float(loss) - float(ref_loss)),
                        "grad_hidden_max_abs_diff": float(mx.max(mx.abs(g_hidden - ref_g_hidden))),
                        "grad_head_max_abs_diff": float(mx.max(mx.abs(g_head - ref_g_head))),
                    }
                )
    return rows


def training_section(log: Logger, vocab: int, steps: int, seed: int = 0) -> dict:
    """Both paths train the same toy model and must reach the same loss."""
    results = {}
    for label, chunk_size in (("naive", None), ("chunked", 512)):
        model = make_model(vocab, D_MODEL, seed=seed)
        task = BigramTask(vocab, n_keys=32, support=64, segment=SEQ_LEN + 1, seed=seed)
        batch = task.sample(BATCH, seed=seed + 1)
        optimizer = optim.AdamW(learning_rate=3e-3)

        def loss_fn(model, tokens, targets, mask):
            hidden = model.hidden(tokens).reshape(-1, D_MODEL)
            flat = targets.reshape(-1)
            if chunk_size is None:
                per_token, _ = naive_linear_cross_entropy(hidden, model.head.weight, flat)
                return per_token
            per_token, _ = chunked_linear_cross_entropy(
                hidden, model.head.weight, flat, chunk_size=chunk_size
            )
            return per_token

        def mean_loss(model, tokens, targets, mask):
            per_token = loss_fn(model, tokens, targets, mask)
            return masked_mean_loss(per_token, mask.reshape(-1))

        value_and_grad = nn.value_and_grad(model, mean_loss)
        history = []
        start = time.perf_counter()
        for step in range(steps + 1):
            loss, grads = value_and_grad(model, batch.tokens, batch.targets, batch.mask)
            mx.eval(loss)
            history.append(float(loss))
            if step < steps:
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
        elapsed = time.perf_counter() - start
        tail = history[max(1, steps - 9) :]
        results[label] = {
            "chunk_size": chunk_size,
            "steps": steps,
            "loss_start": history[0],
            "loss_end": history[-1],
            "loss_tail_mean": statistics.fmean(tail),
            "loss_tail_std": statistics.pstdev(tail),
            "seconds": elapsed,
        }
        log(
            f"  {label:8s} loss {history[0]:8.4f} -> {history[-1]:8.4f} "
            f"(last 10 mean {statistics.fmean(tail):8.4f} +- {statistics.pstdev(tail):.4f}, {elapsed:5.1f}s)"
        )
    results["loss_gap"] = abs(results["naive"]["loss_end"] - results["chunked"]["loss_end"])
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results", help="directory for the result files")
    parser.add_argument("--quick", action="store_true", help="small sweep, for a smoke run")
    parser.add_argument("--steps", type=int, default=40, help="training steps in the liveness check")
    parser.add_argument("--max-vocab", type=int, default=131072)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "run_log.txt")

    vocabularies = [8192, 32768, 65536, 131072]
    vocabularies = sorted(set(v for v in vocabularies if v <= args.max_vocab))
    if args.quick:
        vocabularies = [8192, 32768]
    # (reduction, chunk_size) pairs for the memory sweep.
    chunk_sizes = [(256, "fused"), (1024, "fused"), (4096, "mlx")]
    equality_vocabs = [8192, 32768] if not args.quick else [8192]
    equality_chunks = [64, 256, 1024, 8192] if not args.quick else [64, 1024, 8192]

    log("Cut Cross-Entropy at toy scale")
    log(f"  mlx {mx.__version__} on {platform.platform()} ({platform.machine()})")
    log(f"  default device {mx.default_device()}, metal available {mx.metal.is_available()}")
    log(f"  model: {N_LAYERS} layers, d_model={D_MODEL}, {N_HEADS} heads, d_ff={D_FF}")
    log(f"  tokens per measured pass: {TOKENS} (hidden state pre-computed and detached)")
    log("")

    log("== 1. loss and gradient equality vs the naive path ==")
    equality = equality_section(log, equality_vocabs, equality_chunks)
    worst_loss = max(r["loss_abs_diff"] for r in equality)
    worst_hidden = max(r["grad_hidden_max_abs_diff"] for r in equality)
    worst_head = max(r["grad_head_max_abs_diff"] for r in equality)
    log(f"  worst |loss difference| = {worst_loss:.3e}")
    log(f"  worst d(loss)/d(hidden) max|difference| = {worst_hidden:.3e}")
    log(f"  worst d(loss)/d(head)   max|difference| = {worst_head:.3e}")
    log("")

    log("== 2. peak memory and wall clock vs vocabulary ==")
    memory: list[dict] = []
    header = f"  {'|V|':>8} {'path':>18} {'peak MB':>10} {'active MB':>10} {'ms':>9}  vs naive peak"
    log(header)
    for vocab in vocabularies:
        model = make_model(vocab, D_MODEL, seed=vocab)
        hidden, targets = make_batch(D_MODEL, vocab, TOKENS, seed=vocab)
        head = model.head.weight

        def naive_fn(h, c, targets=targets):
            return naive_linear_cross_entropy(h, c, targets)[0]

        naive_stats = measure(naive_fn, hidden, head)
        naive_stats.update({"vocab": vocab, "path": "naive", "chunk_size": None, "reduction": None})
        memory.append(naive_stats)
        log(
            f"  {vocab:>8} {'naive (materialise)':>18} {naive_stats['peak_mb']:>10.2f} "
            f"{naive_stats['active_mb']:>10.2f} {naive_stats['ms']:>9.2f}  1.00x"
        )
        for chunk_size, reduction in chunk_sizes:
            def chunked_fn(h, c, chunk_size=chunk_size, reduction=reduction, targets=targets):
                return chunked_linear_cross_entropy(
                    h, c, targets, chunk_size=chunk_size, reduction=reduction
                )[0]

            stats = measure(chunked_fn, hidden, head)
            stats.update(
                {
                    "vocab": vocab,
                    "path": "chunked",
                    "chunk_size": chunk_size,
                    "reduction": reduction,
                    "naive_peak_mb": naive_stats["peak_mb"],
                }
            )
            memory.append(stats)
            label = f"chunked/{reduction[:4]}/{chunk_size}"
            log(
                f"  {vocab:>8} {label:>18} {stats['peak_mb']:>10.2f} "
                f"{stats['active_mb']:>10.2f} {stats['ms']:>9.2f}  "
                f"{stats['peak_mb'] / naive_stats['peak_mb']:.2f}x"
            )
    log("")

    scaling: dict[str, dict] = {}
    keys = ["naive"] + [f"chunked/{red}/{cs}" for cs, red in chunk_sizes]
    for path_key in keys:
        if path_key == "naive":
            series = [(r["vocab"], r["peak_mb"]) for r in memory if r["path"] == "naive"]
        else:
            _, reduction, chunk_size = path_key.split("/")
            series = [
                (r["vocab"], r["peak_mb"])
                for r in memory
                if r["path"] == "chunked"
                and r["chunk_size"] == int(chunk_size)
                and r["reduction"] == reduction
            ]
        if len(series) < 2:
            continue
        v_lo, p_lo = series[0]
        v_hi, p_hi = series[-1]
        scaling[path_key] = {
            "vocab_lo": v_lo,
            "vocab_hi": v_hi,
            "peak_lo_mb": p_lo,
            "peak_hi_mb": p_hi,
            "vocab_ratio": v_hi / v_lo,
            "peak_ratio": p_hi / p_lo,
            "log_log_slope": float(
                np.polyfit(np.log([v for v, _ in series]), np.log([p for _, p in series]), 1)[0]
            ),
        }
    for path_key, s in scaling.items():
        log(
            f"  {path_key:>18}: |V| x{s['vocab_ratio']:.1f} -> peak x{s['peak_ratio']:.2f} "
            f"(log-log slope {s['log_log_slope']:.2f})"
        )
    log("")

    log("== 3. gradient filtering (paper Section 4.2) ==")
    filter_vocab = 32768
    model = make_model(filter_vocab, D_MODEL, seed=filter_vocab)
    head = model.head.weight
    hidden, targets = make_batch(D_MODEL, filter_vocab, TOKENS, seed=filter_vocab)

    def chunked_of(threshold):
        def fn(h, c):
            return chunked_linear_cross_entropy(
                h, c, targets, chunk_size=256, grad_threshold=threshold
            )[0]

        return fn

    base_loss, (base_h, base_w) = mx.value_and_grad(chunked_of(0.0), argnums=(0, 1))(hidden, head)
    mx.eval(base_loss, base_h, base_w)
    filtering = []
    for threshold in (0.0, 1e-8, 1e-6, BFLOAT16_EPSILON, 1e-3):
        loss, (g_h, g_w) = mx.value_and_grad(chunked_of(threshold), argnums=(0, 1))(hidden, head)
        mx.eval(loss, g_h, g_w)
        report = last_filter_report()
        row = {
            "threshold": threshold,
            "loss_abs_diff": abs(float(loss) - float(base_loss)),
            "grad_hidden_max_abs_diff": float(mx.max(mx.abs(g_h - base_h))),
            "grad_head_max_abs_diff": float(mx.max(mx.abs(g_w - base_w))),
            "skipped_block_fraction": report.get("skipped_block_fraction", 0.0),
        }
        if threshold == 0.0:
            row["grad_hidden_max_abs_diff"] = 0.0
            row["grad_head_max_abs_diff"] = 0.0
        filtering.append(row)
        log(
            f"  threshold={threshold:>10.3e}  loss diff={row['loss_abs_diff']:.2e}  "
            f"grad(hidden) diff={row['grad_hidden_max_abs_diff']:.2e}  "
            f"grad(head) diff={row['grad_head_max_abs_diff']:.2e}  "
            f"blocks skipped={row['skipped_block_fraction'] * 100:5.1f}%"
        )
    log("")

    log("== 4. training liveness check (both paths must train) ==")
    training = training_section(log, vocab=8192, steps=args.steps)
    log(f"  final loss gap between paths: {training['loss_gap']:.5f}")
    log("")

    payload = {
        "metadata": {
            "mlx_version": mx.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "default_device": str(mx.default_device()),
            "model": {
                "n_layers": N_LAYERS,
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "d_ff": D_FF,
                "seq_len": SEQ_LEN,
                "batch": BATCH,
                "tokens_per_pass": TOKENS,
            },
            "params": {
                "vocabularies": vocabularies,
                "chunk_sizes": chunk_sizes,
                "equality_vocabularies": equality_vocabs,
                "equality_chunk_sizes": equality_chunks,
                "training_vocab": 8192,
                "training_steps": args.steps,
            },
        },
        "equality": equality,
        "memory": memory,
        "scaling": scaling,
        "filtering": filtering,
        "training": training,
    }
    (out_dir / "memory_vs_vocab.json").write_text(json.dumps(payload, indent=2) + "\n")
    log(f"wrote {out_dir / 'memory_vs_vocab.json'}")
    log.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
