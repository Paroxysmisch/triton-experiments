import triton
import triton.language as tl
import torch
import math

# Triton kernel to compute the reciprocal of the square root (rsqrt) of elements
@triton.jit
def rsqrt_kernel(inp_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp_ptr + offsets, mask=mask).to(tl.float32)
    rsqrt_val = tl.rsqrt(x)
    tl.store(out_ptr + offsets, rsqrt_val, mask=mask)

# Function to launch the Triton kernel that computes rsqrt element-wise
def rsqrt_triton(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    block_size = triton.next_power_of_2(min(n_elements, 1024))
    grid = ( (n_elements + block_size - 1) // block_size, )
    rsqrt_kernel[grid](x, out, n_elements, block_size)
    return out

# Wrapper function that performs tensordot on a and b over dims, then applies rsqrt
def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    dot_result = torch.tensordot(a, b, dims=dims)
    return rsqrt_triton(dot_result)
