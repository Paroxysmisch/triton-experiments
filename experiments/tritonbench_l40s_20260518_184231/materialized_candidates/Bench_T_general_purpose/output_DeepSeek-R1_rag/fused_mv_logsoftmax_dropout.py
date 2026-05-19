import triton
import triton.language as tl

@triton.jit
def fused_mv_logsoftmax_dropout_kernel(
    input_ptr, vec_ptr, output_ptr,
    n, m,  # Input matrix dimensions (n x m)
    p, training,
    input_row_stride, input_col_stride,
    vec_stride,
    output_row_stride,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
):
    row_idx = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_idx < n

    z = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Matrix-vector multiplication with tiling
    for col_block in range(0, m, BLOCK_M):
        col_offsets = col_block + tl.arange(0, BLOCK_M)
        col_mask = col_offsets < m

        input_offsets = row_idx[:, None] * input_row_stride + col_offsets[None, :] * input_col_stride
        input_row = tl.load(input_ptr + input_offsets, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        vec_val = tl.load(vec_ptr + col_offsets * vec_stride, mask=col_mask, other=0.0)
        z += tl.sum(input_row * vec_val, axis=1)

    # Log-softmax
    max_z = tl.max(z, axis=0)
    z -= max_z
    sum_exp = tl.sum(tl.exp(z), axis=0)
    log_softmax = z - tl.log(sum_exp)

    # Dropout
    if training:
        rand = tl.rand(row_idx, seed=42)
        mask = rand > p
        scale = 1.0 / (1.0 - p)
        output = tl.where(mask, log_softmax * scale, 0.0)
    else:
        output = log_softmax

    # Store output
    tl.store(output_ptr + row_idx * output_row_stride, output, mask=row_mask)

import torch

def fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None):
    # Shape checks
    assert input.dim() == 2, "Input must be 2D matrix"
    assert vec.dim() == 1, "Vector must be 1D"
    assert input.size(1) == vec.size(0), "Incompatible dimensions"
    assert dim in (0, -1), "Invalid dimension for log_softmax"

    n, m = input.shape
    output = torch.empty(n, device=input.device, dtype=input.dtype) if not inplace else input

    # Configure kernel grid and block sizes
    BLOCK_N = 128  # Adjust based on hardware constraints
    BLOCK_M = 64
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_N']),)

    fused_mv_logsoftmax_dropout_kernel[grid](
        input, vec, output,
        n, m, p, training,
        input.stride(0), input.stride(1),
        vec.stride(0),
        output.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M
    )
    return output if not inplace else input
