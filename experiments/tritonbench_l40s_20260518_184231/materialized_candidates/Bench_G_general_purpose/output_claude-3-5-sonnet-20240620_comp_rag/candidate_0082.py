import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

@triton.jit
def load_reduce_kernel(
    x_ptr,
    y_ptr,
    stride_xm,
    stride_xn,
    stride_y,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Create block pointers
    block_start = pid * BLOCK_M
    offsets = block_start + tl.arange(0, BLOCK_M)
    mask = offsets < M
    x_ptrs = x_ptr + offsets[:, None] * stride_xm + tl.arange(0, BLOCK_N)[None, :] * stride_xn
    
    # Load data
    x = tl.load(x_ptrs, mask=mask[:, None] & (tl.arange(0, BLOCK_N)[None, :] < N), other=-float('inf'))
    
    # Compute row-wise maximum
    max_val = tl.max(x, axis=1)
    
    # Store result
    tl.store(y_ptr + offsets, max_val, mask=mask)

def load_reduce(x):
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    
    BLOCK_M, BLOCK_N = 32, 128
    grid = (triton.cdiv(M, BLOCK_M),)
    
    load_reduce_kernel[grid](
        x, y,
        x.stride(0), x.stride(1), y.stride(0),
        M, N,
        BLOCK_M, BLOCK_N
    )
    
    return y

# Test the kernel
def test_load_reduce():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cuda')
    y_triton = load_reduce(x)
    y_torch = torch.max(x, dim=1)[0]
    torch.testing.assert_close(y_triton, y_torch, atol=1e-2, rtol=1e-2)
    print("Test passed!")

if __name__ == "__main__":
    test_load_reduce()
