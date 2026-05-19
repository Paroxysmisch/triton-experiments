import torch
import triton
import triton.language as tl
import math

@triton.jit
def cos_func(
    a_ptr,          # Pointer to input tensor
    b_ptr,          # Pointer to output tensor
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Static block size parameter
):
    # Calculate offset for each thread block
    pid = tl.program_id(0)  # Get program ID
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input values using the mask
    a_values = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute cosine (ensure float32 precision)
    b_values = tl.cos(a_values.to(tl.float32))
    
    # Store results back to memory
    tl.store(b_ptr + offsets, b_values, mask=mask)

def cos(x):
    # Input validation
    assert x.is_contiguous(), "Input tensor must be contiguous"
    
    # Prepare output tensor
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Calculate optimal block size (nearest power of 2 to sqrt(n_elements))
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    
    # Calculate grid size to cover all elements
    grid = (triton.cdiv(n_elements, block_size),)
    
    # Launch kernel
    cos_func[grid](
        x,              # Input tensor
        output,         # Output tensor
        n_elements,     # Total elements
        block_size,     # Block size
    )
    
    return output
