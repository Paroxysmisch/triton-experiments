import triton
import triton.language as tl

# Define the block size for computation
BLOCK_SIZE = 1024

@triton.jit
def mul2_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program ID for the current block
    pid = tl.program_id(axis=0)
    
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Compute offsets within the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to prevent out-of-bounds access
    mask = offsets < n_elements
    
    # Load elements from input memory
    input_elements = tl.load(input_ptr + offsets, mask=mask)
    
    # Multiply elements by 2
    output_elements = input_elements * 2
    
    # Store the results back to the output memory
    tl.store(output_ptr + offsets, output_elements, mask=mask)

@triton.jit
def mul2_inplace_kernel(input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program ID for the current block
    pid = tl.program_id(axis=0)
    
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Compute offsets within the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to prevent out-of-bounds access
    mask = offsets < n_elements
    
    # Load elements from input memory
    input_elements = tl.load(input_ptr + offsets, mask=mask)
    
    # Multiply elements by 2
    input_elements *= 2
    
    # Store the results back to the input memory (in-place)
    tl.store(input_ptr + offsets, input_elements, mask=mask)

def triton_mul2(input_tensor):
    # Initialize an output tensor with the same shape as the input
    output_tensor = torch.empty_like(input_tensor)
    
    # Determine the number of elements in the input tensor
    n_elements = input_tensor.numel()
    
    # Calculate the grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch the kernel
    mul2_kernel[grid_size](input_tensor, output_tensor, n_elements, BLOCK_SIZE)
    
    return output_tensor

def triton_mul2_inplace(input_tensor):
    # Determine the number of elements in the input tensor
    n_elements = input_tensor.numel()
    
    # Calculate the grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch the kernel
    mul2_inplace_kernel[grid_size](input_tensor, n_elements, BLOCK_SIZE)
