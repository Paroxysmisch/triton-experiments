import triton
import triton.language as tl
import torch

@triton.jit
def cos_func(
    a_ptr,  # pointer to input tensor
    b_ptr,  # pointer to output tensor
    n_elements,  # number of elements in tensor
    BLOCK_SIZE: tl.constexpr,  # number of elements per block
):
    # Calculate the starting point for this block
    offset = tl.program_id(0) * BLOCK_SIZE
    
    # Create a mask for valid elements
    mask = offset + tl.arange(0, BLOCK_SIZE) < n_elements
    
    # Load input values using the mask
    x = tl.load(a_ptr + offset, mask=mask)
    
    # Compute cosine
    output = tl.cos(x)
    
    # Store results
    tl.store(b_ptr + offset, output, mask=mask)

def cos(A):
    # Get input size
    n_elements = A.numel()
    
    # Allocate output tensor
    B = torch.empty_like(A)
    
    # Define block size (can be tuned for performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    cos_func[grid](
        A,  # input tensor
        B,  # output tensor
        n_elements,  # total number of elements
        BLOCK_SIZE,  # elements per block
    )
    
    return B
