"""Fused Metal kernels for the vocabulary reductions.

Both kernels follow the blocking strategy of the paper's Algorithms 2 and 3:
a threadgroup owns one *token* (or one vocabulary position) and walks the
vocabulary (or the batch) inside the kernel, so the only thing that reaches
global memory is the reduced statistic. The logits themselves live in registers
and threadgroup memory, which is why the peak memory of the chunked path is
``O(N + V)`` rather than ``O(N * V)``.

``row_stats`` is a single-pass, numerically stable reduction: each thread keeps
its own running ``(max, sum-of-exp)`` pair over a strided subset of the
vocabulary chunk and the threadgroup merges the partials with the online
log-sum-exp rule. Merging summaries with the online rule gives the same
``(max, sum)`` as one thread scanning the whole chunk, so the two-pass
"global max then sum of exps" formulation is not needed.

``col_sum`` is used by the backward pass and by gradient filtering; its
semantics are exactly ``logits.sum(axis=0)``.
"""

from __future__ import annotations

import mlx.core as mx

# One thread per vocabulary column would waste the GPU on wide blocks, so
# threads stride over the block instead; 1024 threads is Metal's threadgroup
# maximum and caps the block size these kernels accept.
LSE_THREADGROUP = 1024

ROW_STATS_SOURCE = r"""
    // One threadgroup per batch row. The row of the vocabulary chunk is reduced
    // to (max, sum of exp) without ever leaving the threadgroup.
    uint row      = threadgroup_position_in_grid.x;
    uint tid      = thread_position_in_threadgroup.x;
    uint nthreads = threads_per_threadgroup.x;
    uint n        = logits_shape[1];

    float m = -INFINITY;
    float s = 0.0f;
    for (uint j = tid; j < n; j += nthreads) {
        float x = (float)logits[row * n + j];
        if (metal::isfinite(x)) {
            if (x > m) {
                s = s * metal::exp(m - x) + 1.0f;
                m = x;
            } else {
                s += metal::exp(x - m);
            }
        }
    }

    threadgroup float sh_m[1024];
    threadgroup float sh_s[1024];
    sh_m[tid] = m;
    sh_s[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float m1 = sh_m[tid];
            float s1 = sh_s[tid];
            float m2 = sh_m[tid + stride];
            float s2 = sh_s[tid + stride];
            float mm = metal::max(m1, m2);
            float ss = 0.0f;
            // A threadgroup that saw no finite logit contributes nothing at
            // all, which keeps the all -inf row at (-inf, 0) instead of nan.
            if (metal::isfinite(m1)) ss += s1 * metal::exp(m1 - mm);
            if (metal::isfinite(m2)) ss += s2 * metal::exp(m2 - mm);
            sh_m[tid] = mm;
            sh_s[tid] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        row_max[row] = sh_m[0];
        row_sumexp[row] = sh_s[0];
    }
"""

COL_SUM_SOURCE = r"""
    // One threadgroup per vocabulary position. Sums down the batch dimension.
    uint col      = threadgroup_position_in_grid.x;
    uint tid      = thread_position_in_threadgroup.x;
    uint nthreads = threads_per_threadgroup.x;
    uint n        = values_shape[0];

    float part = 0.0f;
    for (uint i = tid; i < n; i += nthreads) {
        part += (float)values[i * values_shape[1] + col];
    }

    threadgroup float sh[1024];
    sh[tid] = part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) col_sums[col] = sh[0];
"""

_ROW_STATS_KERNEL = mx.fast.metal_kernel(
    name="cce_row_stats",
    input_names=["logits"],
    output_names=["row_max", "row_sumexp"],
    source=ROW_STATS_SOURCE,
)

_COL_SUM_KERNEL = mx.fast.metal_kernel(
    name="cce_col_sum",
    input_names=["values"],
    output_names=["col_sums"],
    source=COL_SUM_SOURCE,
)


def row_stats(logits: mx.array) -> tuple[mx.array, mx.array]:
    """Per-row ``(max, sum of exp)`` of a ``(N, V)`` block, computed on the GPU.

    Returns float32 arrays of shape ``(N,)``. All entries of a row being
    ``-inf`` yields ``(-inf, 0)`` rather than a nan.
    """
    if logits.ndim != 2:
        raise ValueError(f"row_stats expects a 2D block, got shape {logits.shape}")
    if logits.shape[1] == 0:
        raise ValueError("row_stats needs at least one vocabulary column")
    if logits.shape[1] > LSE_THREADGROUP:
        raise ValueError(
            f"row_stats is written for vocabulary blocks of at most "
            f"{LSE_THREADGROUP} columns, got {logits.shape[1]}"
        )
    n = logits.shape[0]
    return tuple(
        _ROW_STATS_KERNEL(
            inputs=[logits],
            template=[("T", logits.dtype)],
            grid=(n * LSE_THREADGROUP, 1, 1),
            threadgroup=(LSE_THREADGROUP, 1, 1),
            output_shapes=[(n,), (n,)],
            output_dtypes=[mx.float32, mx.float32],
        )
    )


def col_sum(values: mx.array) -> mx.array:
    """``values.sum(axis=0)`` for a ``(N, V)`` block, computed on the GPU.

    ``V`` must fit in one threadgroup; the batch dimension is unbounded.
    """
    if values.ndim != 2:
        raise ValueError(f"col_sum expects a 2D block, got shape {values.shape}")
    v = values.shape[1]
    if v == 0:
        raise ValueError("col_sum needs at least one column")
    if v > LSE_THREADGROUP:
        raise ValueError(f"col_sum handles at most {LSE_THREADGROUP} columns, got {v}")
    if values.shape[0] == 0:
        return mx.zeros((v,), mx.float32)
    return _COL_SUM_KERNEL(
        inputs=[values],
        template=[("T", values.dtype)],
        grid=(v * LSE_THREADGROUP, 1, 1),
        threadgroup=(LSE_THREADGROUP, 1, 1),
        output_shapes=[(v,)],
        output_dtypes=[mx.float32],
    )[0]
