import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(
    A, B, C,  # Pointers to tensors
    size,     # Size of the tensors
    BLOCK: tl.constexpr,  # Number of elements per block, known at compile time
):
    # Get program ID for parallel execution
    prog_id = tl.program_id(0)
    
    # Calculate offsets for block-wise processing
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    
    # Create mask for handling edge cases
    mask = offs < size
    
    # Load data with mask to prevent out-of-bounds access
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    
    # Perform addition
    c = a + b
    
    # Store result with mask
    tl.store(C + offs, c, mask=mask)

def custom_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Input validation
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU"
    
    # Initialize output tensor
    c = torch.empty_like(a)
    
    # Calculate size and grid
    size = a.numel()
    BLOCK = 16
    grid = lambda meta: (triton.cdiv(size, BLOCK),)
    
    # Launch kernel
    _add_kernel[grid](
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        size,
        BLOCK,
    )
    
    return c
