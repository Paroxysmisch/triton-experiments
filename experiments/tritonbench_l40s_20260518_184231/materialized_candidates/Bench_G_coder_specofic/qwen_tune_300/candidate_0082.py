import torch
import triton
import triton.language as tl
from torch.testing import assert_close

@triton.jit
def load_reduce_kernel(
    x_ptr, y_ptr,
    stride_xm, stride_xn,
    stride_y,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Define a block pointer for accessing the input matrix
    bp0 = tl.block_ptr(x_ptr, (BLOCK_M, BLOCK_N), (stride_xm, stride_xn), (0, 0))
    # Load a block of data from global memory
    x = tl.load(bp0, boundary_check=(0, 1))
    # Compute the max value across each row
    x = tl.max(x, axis=1)
    # Store the result in global memory
    tl.store(y_ptr + stride_y * tl.arange(0, BLOCK_M), x)

def load_reduce(x, dtype):
    # Define block constants
    BLOCK_M = 128
    BLOCK_N = 128
    # Get the number of rows in the input matrix
    M = x.shape[0]
    # Create an empty output vector
    y = torch.empty((M,), dtype=dtype, device=x.device)
    # Calculate the number of warps needed
    num_warps = 4
    if M <= 8192:
        num_warps = 8
    if M <= 4096:
        num_warps = 16
    # Launch the Triton kernel
    load_reduce_kernel[(M,)](
        x, y,
        x.stride(0), x.stride(1),
        y.stride(0),
        num_warps=num_warps,
        num_stages=1,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return y

# Test the kernel
def test_load_reduce():
    # Generate a random input matrix
    x = torch.randn((16384, 128), dtype=torch.float16, device='cuda')
    # Compute the expected result using PyTorch
    y_ref = x.max(axis=1)
    # Run the Triton kernel
    y = load_reduce(x, dtype=torch.float16)
    # Assert that the results are close
    assert_close(y, y_ref, rtol=1e-2, atol=1e-3, check_dtype=False)
