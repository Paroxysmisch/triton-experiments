import triton
import triton.language as tl

@triton.jit
def repeat_interleave_kernel(
    X_ptr,  # Pointer to the input tensor
    Y_ptr,  # Pointer to the output tensor
    repeats_ptr,  # Pointer to the repeats tensor
    n_elements,  # Number of elements in the input tensor
    dim,  # Dimension to repeat along
    repeats_size,  # Size of the repeats tensor
    stride_X,  # Stride of the input tensor along the specified dimension
    stride_Y,  # Stride of the output tensor along the specified dimension
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    X_offsets = offsets * stride_X
    Y_offsets = offsets * stride_Y

    X_vals = tl.load(X_ptr + X_offsets, mask=mask)
    repeats_vals = tl.load(repeats_ptr + offsets % repeats_size, mask=mask)

    for i in range(repeats_size):
        repeat_mask = (offsets % repeats_size) == i
        repeat_count = tl.load(repeats_ptr + i)
        for _ in range(repeat_count):
            tl.store(Y_ptr + Y_offsets, X_vals, mask=repeat_mask)
            Y_offsets += stride_Y

@triton.jit
def log_softmax_kernel(
    Y_ptr,  # Pointer to the input tensor (repeated tensor)
    Z_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the input tensor
    dim,  # Dimension to apply log-softmax along
    stride_Y,  # Stride of the input tensor along the specified dimension
    stride_Z,  # Stride of the output tensor along the specified dimension
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    Y_offsets = offsets * stride_Y
    Z_offsets = offsets * stride_Z

    Y_vals = tl.load(Y_ptr + Y_offsets, mask=mask)

    max_val = tl.max(Y_vals, axis=0)
    Y_vals = Y_vals - max_val

    exp_vals = tl.exp(Y_vals)
    sum_exp = tl.sum(exp_vals, axis=0)
    log_sum_exp = tl.log(sum_exp)

    Z_vals = Y_vals - log_sum_exp
    tl.store(Z_ptr + Z_offsets, Z_vals, mask=mask)

import torch
import triton
import triton.language as tl

def fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None):
    if dim is None:
        input = input.flatten()
        dim = 0
    if output_size is None:
        output_size = input.size(dim) * repeats.sum()
    if dtype is None:
        dtype = input.dtype
    if out is None:
        out = torch.empty(output_size, dtype=dtype, device=input.device)

    # Triton kernel launch configuration
    BLOCK_SIZE = 1024
    grid = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Repeat Interleave
    repeat_interleave_kernel[grid, BLOCK_SIZE](
        input,  # Pointer to the input tensor
        out,  # Pointer to the output tensor
        repeats,  # Pointer to the repeats tensor
        input.numel(),  # Number of elements in the input tensor
        dim,  # Dimension to repeat along
        repeats.numel(),  # Size of the repeats tensor
        input.stride(dim),  # Stride of the input tensor along the specified dimension
        out.stride(dim),  # Stride of the output tensor along the specified dimension
    )

    # Log-Softmax
    log_softmax_kernel[grid, BLOCK_SIZE](
        out,  # Pointer to the input tensor (repeated tensor)
        out,  # Pointer to the output tensor
        out.numel(),  # Number of elements in the input tensor
        dim,  # Dimension to apply log-softmax along
        out.stride(dim),  # Stride of the input tensor along the specified dimension
        out.stride(dim),  # Stride of the output tensor along the specified dimension
    )

    return out
