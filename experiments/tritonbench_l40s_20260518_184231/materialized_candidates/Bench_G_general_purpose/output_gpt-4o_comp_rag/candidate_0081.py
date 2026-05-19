import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def load_reduce_kernel(
    x_ptr, y_ptr,
    stride_xm, stride_xn,
    stride_y,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Define the block indices
    pid_m = tl.program_id(0)
    
    # Compute row start
    row_start = pid_m * BLOCK_M

    # Create a pointer for the current row
    x_offset = row_start * stride_xm
    x_ptrs = x_ptr + x_offset + tl.arange(0, BLOCK_N) * stride_xn

    # Load the data for the current row
    x = tl.load(x_ptrs, mask=tl.arange(0, BLOCK_N) < stride_xn)

    # Compute the maximum across the row
    row_max = tl.max(x, axis=0)

    # Write the result to the output vector
    y_ptrs = y_ptr + row_start * stride_y
    tl.store(y_ptrs, row_max)

# Define the wrapper function
def load_reduce(x, BLOCK_M=128, BLOCK_N=128):
    # Get the shape of the input matrix
    M, N = x.shape

    # Allocate output vector
    y = torch.empty(M, dtype=x.dtype, device=x.device)

    # Define the strides
    stride_xm, stride_xn = x.stride()
    stride_y = y.stride(0)

    # Launch the Triton kernel
    grid = (triton.cdiv(M, BLOCK_M),)
    load_reduce_kernel[grid](
        x, y,
        stride_xm, stride_xn,
        stride_y,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )

    # Verify correctness using PyTorch
    y_ref = torch.max(x, dim=1).values
    torch.testing.assert_close(y, y_ref)

# Example usage
x = torch.randn(1024, 512, device='cuda')
load_reduce(x)
