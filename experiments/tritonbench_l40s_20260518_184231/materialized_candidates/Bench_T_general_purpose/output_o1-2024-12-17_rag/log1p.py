import torch
import triton
import triton.language as tl

@triton.jit
def _log1p_kernel(
    input_ptr,            # pointer to input tensor
    output_ptr,           # pointer to output tensor
    n_elements,           # total number of elements in the tensor
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.log(1.0 + x)
    tl.store(output_ptr + offsets, y, mask=mask)

def log1p(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    # Flatten tensors to 1D for simplicity
    input_flat = input.flatten()
    out_flat = out.flatten()
    
    n_elements = input_flat.numel()
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    _log1p_kernel[grid](
        input_flat, 
        out_flat, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out.reshape_as(input)
