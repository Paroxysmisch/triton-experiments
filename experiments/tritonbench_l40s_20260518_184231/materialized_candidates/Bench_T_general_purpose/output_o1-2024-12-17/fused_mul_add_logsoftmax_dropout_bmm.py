import torch
import triton
import triton.language as tl


@triton.jit
def _fused_mul_add_kernel(
    x1_ptr, x2_ptr, other_ptr, out_ptr,
    N,  # total number of elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < N

    x1 = tl.load(x1_ptr + block_start, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + block_start, mask=mask, other=0.0)
    oth = tl.load(other_ptr + block_start, mask=mask, other=0.0)
    out = x1 * x2 + oth

    tl.store(out_ptr + block_start, out, mask=mask)


@triton.jit
def _log_softmax_kernel(
    in_ptr, out_ptr,
    batch_stride, feature_stride,
    B, F,  # B = batch size, F = feature size
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row of size F in log-softmax
    row_id = tl.program_id(0)
    if row_id >= B:
        return

    row_offset = row_id * batch_stride
    # We'll do a loop in chunks of BLOCK_SIZE along F
    # Step 1: find row max for numerical stability
    max_val = float('-inf')
    for start in range(0, F, BLOCK_SIZE):
        idxs = start + tl.arange(0, BLOCK_SIZE)
        mask = idxs < F
        x = tl.load(in_ptr + row_offset + idxs * feature_stride, mask=mask, other=float('-inf'))
        max_val = tl.maximum(tl.max(x, where=mask), max_val)

    # Step 2: compute denominator = sum of exp(...)
    sum_exp = 0.0
    for start in range(0, F, BLOCK_SIZE):
        idxs = start + tl.arange(0, BLOCK_SIZE)
        mask = idxs < F
        x = tl.load(in_ptr + row_offset + idxs * feature_stride, mask=mask, other=float('-inf'))
        val = tl.exp(x - max_val)
        sum_exp += tl.sum(val, where=mask)

    # Step 3: write output = input - max - log(sum)
    log_denom = tl.log(sum_exp)
    for start in range(0, F, BLOCK_SIZE):
        idxs = start + tl.arange(0, BLOCK_SIZE)
        mask = idxs < F
        x = tl.load(in_ptr + row_offset + idxs * feature_stride, mask=mask, other=0.0)
        out = x - max_val - log_denom
        tl.store(out_ptr + row_offset + idxs * feature_stride, out, mask=mask)


@triton.jit
def _dropout_kernel(
    in
