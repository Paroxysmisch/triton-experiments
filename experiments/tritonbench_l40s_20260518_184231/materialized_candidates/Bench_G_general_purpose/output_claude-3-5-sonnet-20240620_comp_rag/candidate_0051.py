import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    # Compute the program ID and corresponding offsets
    prog_id = tl.program_id(0)
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    
    # Create mask for bounds checking
    mask = offs < size
    
    # Load data with bounds checking
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    
    # Perform addition
    c = a + b
    
    # Store result with bounds checking
    tl.store(C + offs, c, mask=mask)

def custom_add(a: torch.Tensor, b: torch.Tensor):
    # Input validation
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU"
    
    # Create output tensor
    c = torch.empty_like(a)
    
    # Get tensor size
    size = a.numel()
    
    # Define block size and compute grid
    BLOCK = 16
    grid = (triton.cdiv(size, BLOCK),)
    
    # Launch kernel
    _add_kernel[grid](
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        size,
        BLOCK
    )
    
    return c
