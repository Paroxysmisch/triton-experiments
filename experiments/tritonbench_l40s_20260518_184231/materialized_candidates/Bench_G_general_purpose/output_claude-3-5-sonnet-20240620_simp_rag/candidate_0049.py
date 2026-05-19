import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(
    A,  # pointer to first input vector
    B,  # pointer to second input vector 
    C,  # pointer to output vector
    size,  # size of vectors
    BLOCK: tl.constexpr,  # block size (const at compile time)
):
    # Calculate the absolute position in the computation
    prog_id = tl.program_id(0)
    # Calculate offsets for this program instance
    offsets = prog_id * BLOCK + tl.arange(0, BLOCK)
    # Create a mask to handle boundary conditions
    mask = offsets < size
    
    # Load data using the mask
    a = tl.load(A + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)
    
    # Perform the addition
    c = a + b
    
    # Store the result using the same mask
    tl.store(C + offsets, c, mask=mask)

def custom_add(a: torch.Tensor, b: torch.Tensor):
    """
    Wrapper function to perform element-wise addition of two tensors using Triton
    """
    # Basic input validation
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU"
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.dtype == b.dtype, "Input tensors must have the same dtype"
    
    # Create output tensor
    c = torch.empty_like(a)
    
    # Calculate size and grid
    size = a.numel()
    BLOCK = 16  # Can be tuned for performance
    grid = (triton.cdiv(size, BLOCK),)  # Round up division
    
    # Launch kernel
    _add_kernel[grid](
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        size,
        BLOCK,
    )
    
    return c
