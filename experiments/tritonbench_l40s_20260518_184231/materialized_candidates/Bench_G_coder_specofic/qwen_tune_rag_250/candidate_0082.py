import torch
import triton
import triton.language as tl
from torch.testing import assert_close

@triton.jit
def load_reduce_kernel(
    x_ptr, stride_xm, stride_xn,
    y_ptr, stride_y,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Set up block pointers and load data
    xm = tl.arange(0, BLOCK_M)
    xn = tl.arange(0, BLOCK_N)
    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(M, N), strides=(stride_xm, stride_xn),
                                    offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    x = tl.load(x_block_ptr, boundary_check=(0, 1))
    # Compute the reduction
    x = tl.max(x, axis=1)
    # Store the result
    y_offset = xm * stride_y + 0
    tl.store(y_ptr + y_offset, x)

def load_reduce(x):
    M, N = x.shape
    y = torch.empty((M, ), dtype=torch.float32, device=x.device)
    load_reduce_kernel[(1,)](x, x.stride(0), x.stride(1), y, y.stride(0), M, N,
                              BLOCK_M=triton.next_power_of_2(M), BLOCK_N=triton.next_power_of_2(N))
    return y

# Test the kernel
def test_load_reduce():
    B, M, N = 2, 128, 64
    x = torch.randn((B, M, N), dtype=torch.float16, device='cuda')
    y_tri = load_reduce(x)
    y_torch = x.max(axis=1)
    assert_close(y_tri, y_torch, rtol=1e-2, atol=1e-3, check_dtype=False)
