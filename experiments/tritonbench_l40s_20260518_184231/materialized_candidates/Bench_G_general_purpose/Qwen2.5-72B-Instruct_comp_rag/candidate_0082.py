import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def load_reduce_kernel(
    x_ptr,  # Pointer to the input matrix
    y_ptr,  # Pointer to the output vector
    stride_xm,  # Leading dimension stride for the input matrix
    stride_xn,  # Secondary dimension stride for the input matrix
    stride_y,   # Stride for the output vector
    BLOCK_M: tl.constexpr,  # Block size for rows
    BLOCK_N: tl.constexpr   # Block size for columns
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    
    # Compute the row index for this program
    row = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Initialize the maximum value for each row
    max_val = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    
    # Iterate over the columns in blocks
    for col in range(0, stride_xn, BLOCK_N):
        # Load a block of data
        x_block = tl.load(x_ptr + row * stride_xm + col, mask=row < stride_xm, other=-float('inf'))
        
        # Compute the maximum value for this block
        max_val = tl.max(max_val, x_block, axis=1)
    
    # Store the result in the output vector
    tl.store(y_ptr + row, max_val, mask=row < stride_xm)

# Define the wrapper function to test the kernel
def load_reduce(x: torch.Tensor, y: torch.Tensor):
    # Get the dimensions of the input matrix
    M, N = x.shape
    
    # Get the strides of the input matrix
    stride_xm, stride_xn = x.stride()
    
    # Get the stride of the output vector
    stride_y = y.stride(0)
    
    # Define the block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    
    # Compute the grid size
    grid = (M // BLOCK_M + (M % BLOCK_M != 0),)
    
    # Launch the kernel
    load_reduce_kernel[grid](
        x_ptr=x,  # Input matrix pointer
        y_ptr=y,  # Output vector pointer
        stride_xm=stride_xm,  # Leading dimension stride
        stride_xn=stride_xn,  # Secondary dimension stride
        stride_y=stride_y,  # Output vector stride
        BLOCK_M=BLOCK_M,  # Block size for rows
        BLOCK_N=BLOCK_N  # Block size for columns
    )
    
    # Compare the result with PyTorch's max function
    y_torch, _ = torch.max(x, dim=1)
    assert torch.allclose(y, y_torch), "The results do not match PyTorch's max function"

# Test the function
if __name__ == "__main__":
    # Generate a random matrix
    x = torch.randn(1024, 1024, device='cuda')
    
    # Create an empty output vector
    y = torch.empty(1024, device='cuda')
    
    # Run the reduction
    load_reduce(x, y)
    
    print("Reduction completed successfully")
