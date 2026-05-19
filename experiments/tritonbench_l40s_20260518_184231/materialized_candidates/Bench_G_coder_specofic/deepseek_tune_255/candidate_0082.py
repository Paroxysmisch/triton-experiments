import torch
import triton
import triton.language as tl

# Triton kernel for element-wise reduction
@triton.jit
def load_reduce_kernel(
    x_ptr,
    y_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Create a block pointer
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(BLOCK_M, BLOCK_N),
        strides=(stride_xm, stride_xn),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    # Load a block of data
    x = tl.load(x_block_ptr)
    # Compute the row-wise maxima
    x_max = tl.max(x, axis=1)
    # Store the output
    tl.store(y_ptr + tl.arange(0, BLOCK_M), x_max)

# Function to test the Triton kernel
def test_load_reduce(M, N, dtype_str):
    dtype = {"float16": torch.float16, "float32": torch.float32}[dtype_str]
    # Create random input
    x = torch.randn((M, N), device="cuda", dtype=dtype)
    # Create output tensor
    y = torch.empty((M,), device="cuda", dtype=dtype)
    # Define function signature
    sig = (
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
        int,
        int,
    )
    # Check that the kernel produces the expected output
    load_reduce_kernel[M, 1](x, y, N, 1, N, BLOCK_M=128, BLOCK_N=128, sig=sig)
    y_torch = torch.max(x, dim=1)[0]
    assert torch.allclose(y, y_torch)

# Test the kernel
for M in [16, 64, 128, 256, 512, 1024]:
    for N in [16, 64, 128, 256, 512, 1024]:
        for dtype_str in ["float16", "float32"]:
            test_load_reduce(M, N, dtype_str)
