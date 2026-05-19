import triton
import triton.language as tl

@triton.jit
def load_reduce_kernel(
    x_ptr,  # Pointer to the input matrix
    y_ptr,  # Pointer to the output vector
    stride_xm,  # Leading dimension stride of the input matrix
    stride_xn,  # Secondary dimension stride of the input matrix
    stride_y,   # Stride of the output vector
    BLOCK_M: tl.constexpr,  # Block size for rows
    BLOCK_N: tl.constexpr   # Block size for columns
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Compute the row index for this program instance
    row_start = pid * BLOCK_M
    
    # Create a block pointer for the input matrix
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(BLOCK_M, BLOCK_N),
        strides=(stride_xm, stride_xn),
        offsets=(row_start, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    
    # Load the block of data
    x_block = tl.load(x_block_ptr)
    
    # Compute the row-wise maximum
    row_max = tl.max(x_block, axis=1)
    
    # Create a block pointer for the output vector
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(BLOCK_M,),
        strides=(stride_y,),
        offsets=(row_start,),
        block_shape=(BLOCK_M,),
        order=(0,)
    )
    
    # Store the result
    tl.store(y_block_ptr, row_max)

import torch
import triton
import triton.language as tl

def load_reduce(x: torch.Tensor, y: torch.Tensor, BLOCK_M: int, BLOCK_N: int):
    # Ensure the input tensor is on the GPU
    x = x.cuda()
    y = y.cuda()
    
    # Get the dimensions of the input matrix
    M, N = x.shape
    
    # Ensure the output tensor has the correct shape
    assert y.shape == (M,)
    
    # Define the grid and block dimensions
    grid = (M // BLOCK_M, 1, 1)
    block = (BLOCK_M, BLOCK_N, 1)
    
    # Launch the kernel
    load_reduce_kernel[grid, block](
        x_ptr=x,
        y_ptr=y,
        stride_xm=x.stride(0),
        stride_xn=x.stride(1),
        stride_y=y.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )
    
    # Synchronize to ensure the kernel has finished
    torch.cuda.synchronize()
    
    # Compare the result against PyTorch's max function
    y_ref = torch.max(x, dim=1).values
    assert torch.allclose(y, y_ref), "The results do not match the reference implementation."

# Example usage
if __name__ == "__main__":
    # Generate a random matrix
    M, N = 1024, 1024
    x = torch.randn((M, N), dtype=torch.float32)
    
    # Create an empty output vector
    y = torch.empty((M,), dtype=torch.float32)
    
    # Define block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    
    # Run the reduction
    load_reduce(x, y, BLOCK_M, BLOCK_N)
    
    print("Reduction completed successfully.")
