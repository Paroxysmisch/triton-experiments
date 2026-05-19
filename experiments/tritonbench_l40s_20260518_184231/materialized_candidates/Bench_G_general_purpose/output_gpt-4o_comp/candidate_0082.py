import torch
import triton
import triton.language as tl

@triton.jit
def load_reduce_kernel(x_ptr, y_ptr, stride_xm, stride_xn, stride_y, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Calculate the row index for the current block
    pid_m = tl.program_id(0)
    
    # Define block pointers for input and output
    x_offset = pid_m * BLOCK_M * stride_xm
    y_offset = pid_m * BLOCK_M * stride_y
    
    # Create pointers for the current block
    x_ptrs = x_ptr + x_offset + tl.arange(0, BLOCK_N) * stride_xn
    y_ptrs = y_ptr + y_offset
    
    # Load the block of data from the input matrix
    x = tl.load(x_ptrs, mask=True, other=-float('inf'))
    
    # Compute the maximum value across the row
    row_max = tl.max(x, axis=0)
    
    # Store the result in the output vector
    tl.store(y_ptrs, row_max)

def load_reduce(x, BLOCK_M=128, BLOCK_N=128):
    # Ensure input is a 2D tensor
    assert x.ndim == 2, "Input must be a 2D tensor"
    
    # Get dimensions of the input matrix
    M, N = x.shape
    
    # Create an output vector to store the results
    y = torch.empty((M,), dtype=x.dtype, device=x.device)
    
    # Launch the Triton kernel
    grid = (M // BLOCK_M,)
    load_reduce_kernel[grid](
        x, y,
        x.stride(0), x.stride(1), y.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    
    # Verify the results with PyTorch
    expected = torch.max(x, dim=1).values
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)
    
    return y

# Example usage
if __name__ == "__main__":
    # Create a random matrix
    x = torch.randn(1024, 512, device='cuda')
    
    # Perform the reduction
    y = load_reduce(x)
    
    # Print the result
    print(y)
