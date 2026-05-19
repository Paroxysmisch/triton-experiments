import torch
import triton
import triton.language as tl

@triton.jit
def log1p_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.log(x + 1.0)
    tl.store(output_ptr + offsets, y, mask=mask)

def log1p(input, *, out=None):
    # Check input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input")
        if out.dtype != input.dtype:
            raise TypeError("out tensor must have the same dtype as input")
        # Ensure out is contiguous
        if not out.is_contiguous():
            raise ValueError("out tensor must be contiguous")
    
    n_elements = input.numel()
    if n_elements == 0:
        return out  # handle empty tensor
    
    # Define the kernel grid and block sizes
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the kernel
    log1p_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
