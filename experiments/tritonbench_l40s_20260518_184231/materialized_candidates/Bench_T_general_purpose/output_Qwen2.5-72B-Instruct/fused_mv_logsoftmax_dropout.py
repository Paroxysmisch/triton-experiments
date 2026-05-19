import triton
import triton.language as tl

@triton.jit
def fused_mv_logsoftmax_dropout_kernel(
    input_ptr, vec_ptr, out_ptr, n, m, p, training, dim, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Load the input matrix and vector
    input_block = tl.load(input_ptr + block_start * m, mask=block_start + tl.arange(0, BLOCK_SIZE) < n, other=0.0)
    vec = tl.load(vec_ptr, mask=tl.arange(0, m) < m, other=0.0)

    # Matrix-vector multiplication
    z = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(m):
        z += input_block[tl.arange(0, BLOCK_SIZE)] * vec[i]

    # Log-softmax
    z_max = tl.max(z, axis=0)
    z_exp = tl.exp(z - z_max)
    z_sum = tl.sum(z_exp, axis=0)
    s = z - tl.log(z_sum) - z_max

    # Dropout
    if training:
        mask = tl.rand(tl.float32, (BLOCK_SIZE,)) > p
        s = tl.where(mask, s, 0.0)

    # Write the output
    tl.store(out_ptr + block_start, s, mask=block_start + tl.arange(0, BLOCK_SIZE) < n)

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['n', 'm']
)
@triton.jit
def fused_mv_logsoftmax_dropout_kernel(
    input_ptr, vec_ptr, out_ptr, n, m, p, training, dim, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Load the input matrix and vector
    input_block = tl.load(input_ptr + block_start * m, mask=block_start + tl.arange(0, BLOCK_SIZE) < n, other=0.0)
    vec = tl.load(vec_ptr, mask=tl.arange(0, m) < m, other=0.0)

    # Matrix-vector multiplication
    z = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(m):
        z += input_block[tl.arange(0, BLOCK_SIZE)] * vec[i]

    # Log-softmax
    z_max = tl.max(z, axis=0)
    z_exp = tl.exp(z - z_max)
    z_sum = tl.sum(z_exp, axis=0)
    s = z - tl.log(z_sum) - z_max

    # Dropout
    if training:
        mask = tl.rand(tl.float32, (BLOCK_SIZE,)) > p
        s = tl.where(mask, s, 0.0)

    # Write the output
    tl.store(out_ptr + block_start, s, mask=block_start + tl.arange(0, BLOCK_SIZE) < n)

def fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None):
    if input.dim() != 2 or vec.dim() != 1:
        raise ValueError("Input must be a 2D tensor and vec must be a 1D tensor.")
    if input.size(1) != vec.size(0):
        raise ValueError("The number of columns in input must match the size of vec.")

    n, m = input.size(0), input.size(1)
    if out is None:
        out = torch.empty((n,), dtype=input.dtype, device=input.device)

    if inplace:
        out = input

    grid = (triton.cdiv(n, 128),)
    fused_mv_logsoftmax_dropout_kernel[grid](
        input, vec, out, n, m, p, training, dim, BLOCK_SIZE=128
    )

    return out
