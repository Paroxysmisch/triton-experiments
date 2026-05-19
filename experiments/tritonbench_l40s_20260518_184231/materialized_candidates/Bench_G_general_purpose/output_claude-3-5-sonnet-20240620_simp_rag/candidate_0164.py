import torch
import triton
import triton.language as tl
import math

@triton.jit
def cos_func(
    a_ptr,  # Pointer to input tensor
    b_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Static block size for processing
):
    # Get the absolute position in the computation grid
    pid = tl.program_id(axis=0)
    # Calculate offset for current block
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offset < n_elements
    
    # Load input values using the mask
    x = tl.load(a_ptr + offset, mask=mask)
    
    # Compute cosine (cast to float32 for better precision)
    output = tl.cos(x.to(tl.float32))
    
    # Store results back to memory
    tl.store(b_ptr + offset, output, mask=mask)

def cos(A):
    # Input validation
    assert A.is_contiguous(), "Input tensor must be contiguous"
    
    # Create output tensor with same properties as input
    B = torch.empty_like(A)
    
    # Calculate total elements
    n_elements = A.numel()
    
    # Compute block size (power of 2 for efficiency)
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    
    # Calculate grid size to cover all elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    cos_func[grid](
        A,
        B,
        n_elements,
        BLOCK_SIZE,
    )
    
    return B
