import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(
    A_ptr, B_ptr, C_ptr,
    size,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    a = tl.load(A_ptr + offs, mask=mask)
    b = tl.load(B_ptr + offs, mask=mask)
    c = a + b
    tl.store(C_ptr + offs, c, mask=mask)

def custom_add(a: torch.Tensor, b: torch.Tensor):
    # Ensure inputs are on CUDA and have matching shapes
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"
    assert a.shape == b.shape, "Inputs must have the same shape"
    
    # Ensure contiguous memory layout
    a, b = a.contiguous(), b.contiguous()
    
    # Create output tensor
    c = torch.empty_like(a)
    
    # Total number of elements
    size = a.numel()
    
    # Block size (elements processed per program instance)
    BLOCK = 16
    
    # Grid size (number of program instances)
    grid = (triton.cdiv(size, BLOCK), )
    
    # Launch kernel
    _add_kernel[grid](a.data_ptr(), b.data_ptr(), c.data_ptr(), size, BLOCK=BLOCK)
    
    return c
