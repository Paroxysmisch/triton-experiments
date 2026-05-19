import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 1024  # You can adjust this value based on your GPU architecture and problem size

@triton.jit
def puzzle1_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the program index
    pid = tl.program_id(axis=0)
    
    # Calculate the start and end index for this block
    start = pid * BLOCK_SIZE
    end = tl.min(start + BLOCK_SIZE, n_elements)
    
    # Create a range of indices for the current block
    offsets = start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we do not go out of bounds
    mask = offsets < n_elements
    
    # Load x from global memory
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Add the constant value (10) to each element
    y = x + 10
    
    # Store the result in the output tensor
    tl.store(y_ptr + offsets, y, mask=mask)

def puzzle1(x):
    # Get the number of elements in the input tensor
    n_elements = x.numel()
    
    # Allocate output tensor
    y = torch.empty_like(x)
    
    # Calculate the number of blocks needed
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the kernel
    puzzle1_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return y
