import torch
import triton
import triton.language as tl

# Triton kernel for element-wise addition
@triton.jit
def add_kernel(
    x_ptr,  # Pointer to the first input tensor
    y_ptr,  # Pointer to the second input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensors
    BLOCK_SIZE: tl.constexpr  # Block size for the kernel
):
    # Get the block index
    pid = tl.program_id(axis=0)
    
    # Compute the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Compute the offsets for the elements in this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle out-of-bounds accesses
    mask = offsets < n_elements
    
    # Load the elements from the input tensors
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute the sum
    output = x + y
    
    # Store the result in the output tensor
    tl.store(output_ptr + offsets, output, mask=mask)

# Wrapper function to launch the kernel
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Ensure all tensors are on the CUDA device
    x = x.cuda()
    y = y.cuda()
    
    # Create the output tensor
    output = torch.empty_like(x).cuda()
    
    # Calculate the total number of elements
    n_elements = x.numel()
    
    # Define the block size
    BLOCK_SIZE = 1024
    
    # Define the grid size
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    add_kernel[grid, BLOCK_SIZE](x, y, output, n_elements, BLOCK_SIZE)
    
    return output
