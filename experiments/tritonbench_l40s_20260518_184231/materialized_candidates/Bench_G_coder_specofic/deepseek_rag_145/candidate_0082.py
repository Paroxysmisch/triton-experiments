import triton.language as tl
import torch
import numpy as np

@triton.jit
def load_reduce_kernel(
    X_ptr,
    Y_ptr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_y: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)

    ptr_x = X_ptr + row * stride_xm + col * stride_xn
    ptr_y = Y_ptr + row

    val = tl.load(ptr_x)

    for _ in range(BLOCK_N):
        val = tl.max(val, tl.load(ptr_x))
        ptr_x += BLOCK_N

    tl.store(ptr_y, val)

def load_reduce(x, y):
    X_ptr = tl.pointer(x, tl.float32)
    Y_ptr = tl.pointer(y, tl.float32)

    stride_xm = x.shape[0]
    stride_xn = x.shape[1]
    stride_y = y.shape[0]

    BLOCK_M = 32
    BLOCK_N = 32

    grid = lambda : (x.shape[0], BLOCK_N)

    load_reduce_kernel[grid](X_ptr, Y_ptr, stride_xm, stride_xn, stride_y, BLOCK_M, BLOCK_N)

def assert_close(x, y):
    assert np.allclose(x.numpy(), y.numpy()), f"Expected {y}, got {x}"

def test_load_reduce():
    torch.manual_seed(42)

    x = torch.randn(100, 100)
    y = torch.empty(100)

    load_reduce(x, y)
    assert_close(y, x.max(dim=-1)[0])

test_load_reduce()
