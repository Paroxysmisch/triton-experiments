import torch
import triton
import triton.language as tl

@triton.jit
def load_reduce_kernel(
    x_ptr,
    y_ptr,
    stride_xm,
    stride_xn,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Kernel to load, reduce and store
    mid = tl.arange(0, BLOCK_M)
    nid = tl.arange(0, BLOCK_N)

    block_ptr = tl.make_block_ptr(base=x_ptr, shape=(BLOCK_M, BLOCK_N), strides=(stride_xm, stride_xn), offsets=(mid * BLOCK_M, nid))
    x = tl.load(block_ptr)
    y = tl.max(x, axis=1)[:, None]

    tl.store(y_ptr + mid * stride_y, y)

def test_load_reduce(dtype_n, M, N, BLOCK_M, BLOCK_N):
    load_reduce = triton.JITFunction(load_reduce_kernel.deduce_shapes((M, N), ((M, N), (M,), (N,)))(dtype_n, dtype_n), target='cuda')

    x = torch.randn((M, N), device="cuda", dtype=dtype_n)
    y = torch.empty((M,), device="cuda", dtype=dtype_n)

    load_reduce(x, y, int(x.stride(0)), int(x.stride(1)), int(y.stride(0)), BLOCK_M, BLOCK_N)

    golden = x.max(axis=1)
    assert torch.allclose(y, golden)

toy_backend.assert_close(load_reduce, test_load_reduce)
