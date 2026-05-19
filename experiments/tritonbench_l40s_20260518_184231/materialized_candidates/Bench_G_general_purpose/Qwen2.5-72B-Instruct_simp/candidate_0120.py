import triton
import triton.language as tl

@triton.jit
def chunk_global_cumsum_scalar_kernel(
    s_ptr,  # Pointer to the input tensor
    o_ptr,  # Pointer to the output tensor
    n_elements,  # Total number of elements in the last dimension
    BT,  # Block size
    stride_s,  # Stride of the input tensor in the last dimension
    stride_o,  # Stride of the output tensor in the last dimension
    pid,  # Block ID
):
    block_start = pid * BT
    block_end = min(block_start + BT, n_elements)
    running_total = 0.0

    for i in range(block_start, block_end):
        s_val = tl.load(s_ptr + i * stride_s)
        running_total += s_val
        tl.store(o_ptr + i * stride_o, running_total)

    if pid > 0:
        prev_block_end = (pid - 1) * BT
        prev_block_total = tl.load(o_ptr + (prev_block_end - 1) * stride_o)
        running_total += prev_block_total

        for i in range(block_start, block_end):
            s_val = tl.load(s_ptr + i * stride_s)
            running_total += s_val - s_val
            tl.store(o_ptr + i * stride_o, running_total)

@triton.jit
def chunk_global_cumsum_scalar(
    s_ptr,  # Pointer to the input tensor
    o_ptr,  # Pointer to the output tensor
    n_elements,  # Total number of elements in the last dimension
    BT,  # Block size
    stride_s,  # Stride of the input tensor in the last dimension
    stride_o,  # Stride of the output tensor in the last dimension
    grid,  # Grid size
):
    chunk_global_cumsum_scalar_kernel[grid](
        s_ptr, o_ptr, n_elements, BT, stride_s, stride_o, tl.program_id(0)
    )

import torch

def chunk_global_cumsum(s, block_size):
    assert s.dim() == 3, "Input tensor must be 3D"
    n_elements = s.size(-1)
    s_ptr = triton.core.get_tensor_ptr(s, 'cuda')
    o = torch.empty_like(s)
    o_ptr = triton.core.get_tensor_ptr(o, 'cuda')
    stride_s = s.stride(-1)
    stride_o = o.stride(-1)
    grid = (n_elements + block_size - 1) // block_size

    chunk_global_cumsum_scalar[
        grid
    ](
        s_ptr, o_ptr, n_elements, block_size, stride_s, stride_o, grid
    )

    return o

import torch

# Example input tensor
s = torch.randn(2, 3, 1024, device='cuda')

# Block size
block_size = 256

# Perform chunked global cumulative sum
o = chunk_global_cumsum(s, block_size)

print(o)
