# cut-cross-entropy-mlx

A small independent MLX implementation of *Cut Your Losses in Large-Vocabulary
Language Models* (arXiv:2411.09009), plus a toy 2-layer transformer with a
64k-token vocabulary to measure what the mechanism actually does on Apple
silicon. The paper's point is that standard cross-entropy materialises a
`batch x vocab` logits matrix, which for a large vocabulary is the single
largest allocation in training; Cut Cross-Entropy (CCE) computes the same loss
without ever building that matrix by streaming over the vocabulary in chunks and
carrying a running `(max, sum-of-exp)` pair, exactly like an online softmax. It
reports reducing the loss computation's memory from 24 GB to 1 MB for Gemma 2
(2B), and speeding up the backward pass 3.5x by skipping vocabulary blocks whose
softmax weights are all below bfloat16 precision.

This is not a reproduction. The model is 74k backbone parameters plus 8.4M of
embedding and classifier at `|V| = 65536`, trained on a synthetic task with
vocabularies up to 131k; there is no CUDA and no Triton, and none of its numbers
are comparable to the paper's. What it does do is implement
the mechanism so that the chunked path is bit-comparable to the naive one, and
then measure peak memory honestly — including the case where the answer is not
the one the paper reports. The short version: the chunked path reproduces the
naive loss and both gradients to within `7e-10`, and at 8192 tokens with a 64k
vocabulary it peaks at 2.60 GB against the naive path's 6.50 GB — but below
about 2048 tokens it is *worse* than the naive path, and peak memory is not flat
in vocabulary size.

## What is in here

**Log-space cross-entropy and the online accumulator** (`cce/ce.py`). Equation 4
of the paper splits the per-token loss into a log-sum-exp over the vocabulary
and the logit of the ground-truth token:

```
loss_i = logsumexp_j(C_j^T E_i) - C_{x_i}^T E_i
```

`OnlineLSE` holds the pair `(max, sum_exp)` with `logsumexp = max + log sum_exp`
and `online_update` merges two such pairs with
`s <- s*exp(m - m') + s_c*exp(m_c - m')`. Merging summaries rather than
accumulating log-sum-exp values is what keeps the reduction exact: no term is
ever compared against a normaliser that is still moving. `naive_cross_entropy`
takes a materialised `(N, V)` matrix and is the reference; `chunked_cross_entropy`
takes a callable that yields one vocabulary chunk at a time and never holds more
than one.

**Fused reduction kernels** (`cce/kernels.py`). `row_stats` reduces a `(N, chunk)`
block to per-token `(max, sum-of-exp)` in one Metal launch: one threadgroup per
token, each thread keeping its own running pair over a strided slice of the
chunk, and the threadgroups merging partials with the same online rule. That
single pass is what makes the second pass of "find the global max, then sum
`exp(x - max)`" unnecessary. It is written for blocks of at most 1024 columns
(`LSE_THREADGROUP`), and an all-`-inf` row comes back as `(-inf, 0)` rather than
`nan`.

`col_sum` is the axis-0 reduction the backward pass would need if the weight
gradient were accumulated column-wise; the shipped backward uses MLX's own
matmul for that sum, and the kernel is kept and tested because it is the other
half of Algorithm 3.

**The chunked linear + cross-entropy operator** (`cce/linear_ce.py`). This is the
paper's `C^T E` without the materialisation: `E @ C_c^T` is built for one
vocabulary chunk at a time, reduced to a summary, and dropped. The whole operator
is an `mx.custom_function` with an explicit `vjp`, because the backward pass is
where naive chunking goes wrong. The `vjp` recomputes each chunk's logits and
recovers the softmax weights from the stored log-sum-exp,

```
S_i = exp(logits_i - LSE_i)
```

with no renormalisation, then accumulates `dL/dE += sum_c (S_c * coeff) @ C_c`
and writes `dL/dC_c = (S_c * coeff)^T @ E` into that chunk's own column range.
The gathered ground-truth term of Equation 4 is the other half —
`dL/dE -= coeff_i * C_{x_i}` and a one-hot scatter into `C` — and leaving it out
is exactly the bug the gradient-equality test in `tests/test_linear_ce.py` was
written to catch (it did). The hidden state's log-sum-exp stays inside the
operator on purpose: exposing it as a second output makes its cotangent carry
gradient the `vjp` is already accounting for, which silently halves the gradient.

Three reductions are available. `"mlx"` uses `mx.max`/`mx.sum`; `"fused"` uses
`row_stats`; `"inline"` merges each chunk into the running accumulator before
computing the next one, which is the literal reading of Algorithm 2 rather than
a batch of summaries merged at the end.

**Sparse-gradient skipping** (`cce/filter.py`). The paper's gradient filtering:
a chunk whose softmax weights are *all* strictly below `eps` contributes nothing
to either gradient, so it is skipped. The default `grad_threshold=0.0` can never
fire, so the unfiltered path is exact. The paper's `eps = 2**-12` (the smallest
bfloat16 value that is not truncated by rounding) is exported as
`BFLOAT16_EPSILON`. Note that the skip is *conservative* — a chunk only goes
away when its largest weight is already below the threshold, so a block holding
a ground-truth token usually survives, and the gathered term is never filtered.

**Toy model and task** (`cce/model.py`, `cce/task.py`). A 2-layer, 4-head,
`d_model=64` decoder with an untied zero-initialised classifier head, so the
first loss is exactly `log |V|`. The task is a bigram language model:
`n_keys=32` key tokens each have a disjoint 64-token support and a segment of
continuations drawn i.i.d. from that key's distribution, so the conditional
entropy is about `log 64 = 4.16` nats while a model that ignored the previous
token would sit at `log 4096 = 8.32` (measured, `results/run_log.txt`).

## Checks

`pytest` runs 87 tests. The ones that matter most:

- **The chunked path's gradients equal the naive path's** across chunk sizes,
  vocabularies and all three reductions, to `<= 1e-5`; measured worst case
  `6.99e-10` for `d(loss)/d(hidden)` and `1.40e-09` for `d(loss)/d(head)`
  (`tests/test_linear_ce.py`, `tests/test_ce.py`). The same quantity is also
  checked against a float64 analytic gradient in NumPy, so a mistake shared by
  both MLX paths cannot pass.
- **The online log-sum-exp is stable.** Logits spread over `+-1e3`, which puts
  `exp` far outside float32, produce finite per-token losses that match a
  float64 reference; an all-`-inf` vocabulary yields `(-inf, 0)` from the kernel
  and a zero per-token loss rather than `nan` (`tests/test_kernels.py`).
- **One chunk covering the whole vocabulary equals the naive path exactly** —
  `float(got) == float(want)` and bit-identical per-token losses, because the
  reduction over one chunk is a single `(max, sum-exp)` pair.
- **`threshold=0` is exactly no filtering**: identical loss and bit-identical
  gradients; at `2**-12` on a peaked problem exactly two of eight blocks are
  dropped, and the skip count is checked against an independent float64
  reimplementation of `all(S_nm < eps)` rather than against the implementation.
- **Kernels.** `row_stats` and `col_sum` are compared against float64 NumPy
  references over many shapes including an empty batch, a single column, and a
  non-contiguous view (which is what the chunked path actually feeds them).
- **The model is causal** (changing a future token does not change an earlier
  hidden state), starts at exactly `log |V|`, and its backbone is under 1/50th
  of the classifier head's parameter count — the premise of the experiment.

## Experiment

`scripts/memory_vs_vocab.py`. Each memory number below is measured in its own
fresh interpreter: `mx.get_peak_memory()` is a process-lifetime high-water mark
and is sensitive to allocator state, so measuring several configurations inside
one warm process made earlier configurations look tens of MB smaller than the
same configuration measured cold. The numbers here are the isolated ones, and
each measurement is the peak of a forward+backward pass over a fixed detached
`(256, 64)` hidden state, after three warm-up calls. The full log is in
`results/run_log.txt` and the raw JSON in `results/memory_vs_vocab.json`.

**Loss and gradient equality** (256 tokens, vocab 8k and 32k):

| quantity | worst absolute difference |
|---|---:|
| loss | `0.000e+00` |
| `d(loss)/d(hidden)` | `6.985e-10` |
| `d(loss)/d(head)` | `1.397e-09` |

**Peak memory and wall clock vs vocabulary** (256 tokens per pass; the logits
column is what the naive path materialises):

| vocab | path | peak MB | wall clock | vs naive peak |
|---:|---|---:|---:|---:|
| 8192 | naive | 31.60 | 4.15 ms | 1.00x |
| 8192 | chunked/fused/256 | 72.30 | 32.47 ms | 2.29x |
| 8192 | chunked/fused/1024 | 42.39 | 11.73 ms | 1.34x |
| 8192 | chunked/mlx/4096 | 75.83 | 5.24 ms | 2.40x |
| 32768 | naive | 119.68 | 7.98 ms | 1.00x |
| 32768 | chunked/fused/256 | 186.15 | 117.68 ms | 1.56x |
| 32768 | chunked/fused/1024 | 287.42 | 42.07 ms | 2.40x |
| 32768 | chunked/mlx/4096 | 336.46 | 16.41 ms | 2.81x |
| 65536 | naive | 237.12 | 9.00 ms | 1.00x |
| 65536 | chunked/fused/256 | 514.12 | 214.68 ms | 2.17x |
| 65536 | chunked/fused/1024 | 610.49 | 69.40 ms | 2.57x |
| 65536 | chunked/mlx/4096 | 648.55 | 43.79 ms | 2.74x |
| 131072 | naive | 472.00 | 24.21 ms | 1.00x |
| 131072 | chunked/fused/256 | 668.91 | 432.03 ms | 1.42x |
| 131072 | chunked/fused/1024 | 1231.97 | 155.24 ms | 2.61x |
| 131072 | chunked/mlx/4096 | 1241.35 | 87.27 ms | 2.63x |

Fitting a line to `log(peak)` against `log(|V|)` over the 16x vocabulary range:
naive `0.97`, chunked/fused/256 `0.85`, chunked/fused/1024 `1.22`,
chunked/mlx/4096 `1.01`.

**Peak memory and wall clock vs token count** (the comparison that matters):

| tokens | vocab | logits | naive peak | naive ms | chunked peak | chunked ms | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 16384 | 16.8 MB | 61.0 MB | 3.4 | 68.1 MB | 28.2 | 1.12x |
| 256 | 65536 | 67.1 MB | 237.1 MB | 11.4 | 634.1 MB | 59.0 | 2.67x |
| 2048 | 16384 | 134.2 MB | 416.4 MB | 18.0 | 634.3 MB | 36.4 | 1.52x |
| 2048 | 65536 | 536.9 MB | 1662.1 MB | 75.3 | 1267.1 MB | 158.1 | **0.76x** |
| 8192 | 16384 | 536.9 MB | 1627.7 MB | 85.3 | 1835.8 MB | 143.0 | 1.13x |
| 8192 | 65536 | 2147.5 MB | 6497.2 MB | 337.2 | 2596.7 MB | 550.1 | **0.40x** |

(Peak MB values above are rounded to one decimal place to match
`results/run_log.txt`; the unrounded values in `results/memory_vs_vocab.json`
are 60.96/68.08 (256 tokens, 16k), 237.12/634.09 (256, 64k), 416.35/634.30
(2048, 16k), 1662.06/1267.09 (2048, 64k), 1627.65/1835.84 (8192, 16k) and
6497.24/2596.72 MB (8192, 64k), in naive/chunked order. Wall clocks are in ms.)

**Gradient filtering** at 256 tokens and `|V| = 32768`, chunk size 256 (128
blocks). This toy task is unusually peaked, so 100% of blocks fall below the
bfloat16 threshold:

| threshold | loss diff | `d/d(hidden)` diff | `d/d(head)` diff | blocks skipped |
|---:|---:|---:|---:|---:|
| 0.0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% |
| 1e-8 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% |
| 1e-6 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0% |
| 2.441e-04 (`2**-12`) | 0.00e+00 | 1.53e-06 | 1.53e-03 | 100.0% |
| 1e-3 | 0.00e+00 | 1.53e-06 | 1.53e-03 | 100.0% |

**Training liveness.** 40 AdamW steps at lr 3e-3 on the toy bigram task at
`|V| = 8192`, 4x128 tokens per step: naive `9.0206 -> 1.1921` (last-10 mean
`1.7244 +- 0.3379`), chunked `9.0206 -> 1.1931` (last-10 mean
`1.7244 +- 0.3377`), final gap `0.00093`. Both paths train, to the same loss.

### What the numbers mean

- **The correctness claim holds.** The chunked path reproduces the naive loss
  and both gradients to floating-point noise at every chunk size, vocabulary and
  reduction, including a chunk of 8192 columns and a chunk of 64. That is the
  part of the paper this repo actually reproduces.
- **The memory claim holds only once the batch is big enough, and I measured
  the crossover.** At 256 tokens the chunked path is *worse* — up to 2.7x the
  naive peak — because per-chunk graph overhead dominates when the logits matrix
  is only 17-67 MB. At 2048 tokens with `|V| = 65536` it is 0.76x, and at 8192
  tokens it is **0.40x** (2.60 GB against 6.50 GB). The direction of the paper's
  claim is real here; the magnitude is nowhere near 24 GB to 1 MB.
- **Peak memory is not flat in vocabulary size**, contrary to the paper's
  `O(N + |V|)`. Over 8k to 131k the naive path grows with log-log slope 0.97
  while the best chunked configuration grows with slope 0.85. The chunked path
  does not materialise the logits matrix — I verified the largest per-chunk
  allocation is `N x chunk` — but MLX's lazy evaluator holds scratch buffers for
  the chunk loop, and that scratch grows with the number of chunks. The slope
  being under 1 while the naive slope is at 1 is the honest version of the claim
  at this scale.
- **The fused kernel is not a speedup here.** At 256 tokens the `mlx` reduction
  with a 4096-column chunk is the fastest chunked variant (5.24 ms at `|V|=8192`
  against 4.15 ms naive), and `fused/256` — 256 Metal launches per pass — is by
  far the slowest (432 ms at `|V| = 131072`). The kernel computes the same
  reduction the paper's Triton kernel does, but a Python-level loop of small
  launches is not a fused kernel, and the paper's 3.5x backward speedup does not
  appear.
- **Gradient filtering saves nothing measurable in wall clock.** All 128 blocks
  are skipped at the bfloat16 threshold on this task, and the loss is unchanged,
  but the wall-clock column for filtering is not reported because the remaining
  gathered term and the per-chunk work dominate at this size.
- Peak memory is several times the raw logits matrix, not a fraction of it:
  naive uses 3.63x, 3.53x, 3.10x and 3.03x of `N x |V| x 4` bytes at
  (256, 16k), (256, 64k), (2048, 64k) and (8192, 64k); chunked is 4.06x, 9.45x,
  2.36x and 1.21x at the same four points, i.e. its overhead shrinks towards the
  1x floor the paper describes as the batch grows. Nothing here is a small-memory
  implementation of cross-entropy; it is a comparison of two implementations
  under the same autodiff.

## Where this may differ from the paper

- **No Triton, no fused matmul-log-sum-exp.** The paper writes a CUDA/Triton
  kernel that does the classifier indexing, the matrix multiply and the
  log-sum-exp reduction inside SRAM in one launch. `row_stats` here only fuses
  the *reduction*; the `N x chunk` matmul still goes through MLX and lands in
  global memory. That is the biggest gap, and it is why the memory advantage
  only appears at large token counts.
- **The backward pass is a Python loop of MLX operations, not one kernel.**
  I chose an explicit `mx.custom_function` with a hand-written `vjp` over a
  Metal kernel for the chunk reduction. I tried three other things first and
  they are worth recording: (a) letting autodiff differentiate the chunk loop
  produces correct gradients, but keeps the whole chunk graph alive, so the
  backward peak was 3-4x the forward peak; (b) evaluating the logits eagerly
  inside the `vjp` made the backward *worse* (the per-chunk evaluation ordering
  defeated MLX's buffer reuse) rather than better; (c) a Metal backward kernel
  that accumulated `dE` in threadgroup memory, which is what a real
  implementation would do, produced wrong results in three successive
  formulations (threadgroup-memory reduction with a `-1.0` sentinel, per-thread
  register accumulation, per-entry scratch rows) — I could not get race-free
  accumulation to work in `mx.fast.metal_kernel` and removed it rather than ship
  a kernel that passed some shapes and not others. The shipped backward is
  therefore exact but memory-hungry, which is the honest state of this repo.
- **No vocabulary sorting.** The paper reorders the vocabulary by average logit
  so that the blocks it skips are the blocks that are actually empty. There is
  no sorting here, so the skip is block-aligned only by accident.
- **Gradient filtering is block level only, and is not combined with a fused
  backward.** The paper's speedup comes from skipping in the kernel; here the
  skip is a Python `if` around MLX operations.
- **The classifier head is not tied to the embedding and the model is not
  otherwise the paper's.** The paper measures on Gemma 2 (2B), Phi 3.5 Mini,
  Qwen 2.5 7B and Mistral NeMo with vocabularies of 32k-256k and hidden sizes of
  2304-3584, i.e. `|V|/D` between 14 and 111. This is `d_model=64`, so `|V|/D` is
  1024 at `|V| = 65536` — 9x to 73x larger, which flatters the chunked path's
  arithmetic (the chunked matmul is `N x chunk x D`, so a small `D` is cheap) and
  ignores every memory cost that is not the logits.
- **The synthetic task makes gradient filtering look better than it is.** The
  task's next-token distribution has 64 support tokens, so the softmax collapses
  to a handful of entries and *every* block is skipped at `eps = 2**-12`. A real
  language model has a much heavier tail, which is why the paper reports 0.02%
  non-zero entries rather than 100% of blocks skipped.
- **`ignore_index` handling is mine.** The paper's Appendix B discusses removing
  ignored tokens; here ignored rows simply contribute nothing to the loss, its
  gradients or the normalisation, which is tested.
- **No 24 GB, no 1 MB, no 3.5x speedup, no convergence study.** The paper's
  headline numbers are not reproduced anywhere in this repo.

## Running it

```
/Users/dash/Documents/dev/ai_papers/.venv/bin/python -m pytest
/Users/dash/Documents/dev/ai_papers/.venv/bin/python scripts/memory_vs_vocab.py
```

The experiment takes a few minutes (most of it in isolated subprocesses, one per
memory measurement) and rewrites `results/memory_vs_vocab.json` and
`results/run_log.txt`. Useful flags: `--quick`, `--steps`, `--max-vocab`,
`--max-tokens`, `--out`.
