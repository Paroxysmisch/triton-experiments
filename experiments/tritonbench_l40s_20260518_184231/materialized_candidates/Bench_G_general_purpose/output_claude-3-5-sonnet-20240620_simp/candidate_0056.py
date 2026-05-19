import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(
    A, # Pointer to first input tensor
    B, # Pointer to second input tensor
    C, # Pointer to output tensor
    size, # Size of the tensors
    BLOCK: tl.constexpr, # Number of elements per block, as compile-time constant
):
    # Get the program ID
    prog_id = tl.program_id(0)
    
    # Calculate offsets for this program instance
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    
    # Create a mask for bounds checking
    mask = offs < size
    
    # Load data with bounds checking
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    
    # Perform addition
    c = a + b
    
    # Store result with bounds checking
    tl.store(C + offs, c, mask=mask)

def custom_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Input validation
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU"
    
    # Get total size of tensors
    size = a.numel()
    
    # Create output tensor
    c = torch.empty_like(a)
    
    # Define block size
    BLOCK = 16
    
    # Calculate grid size
    grid = (size + BLOCK - 1) // BLOCK
    
    # Launch kernel
    _add_kernel[grid](
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        size,
        BLOCK,
    )
    
    return c
