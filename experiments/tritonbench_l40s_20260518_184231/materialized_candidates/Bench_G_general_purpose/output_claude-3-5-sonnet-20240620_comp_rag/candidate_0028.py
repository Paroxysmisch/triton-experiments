import torch
import triton
import triton.language as tl

@triton.jit
def _dropout(
    x_ptr,          # pointer to input tensor
    x_keep_ptr,     # pointer to mask tensor
    output_ptr,     # pointer to output tensor
    n_elements,     # number of elements in the tensor
    p,              # dropout probability
    BLOCK_SIZE: tl.constexpr,  # size of parallel blocks
):
    # Compute memory offsets for this program instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for bounds checking
    mask = offsets < n_elements
    
    # Load data and mask
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    
    # Apply dropout: scale by 1/(1-p) where kept, 0 otherwise
    output = tl.where(x_keep, x / (1 - p), 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def dropout(x, x_keep, p):
    # Ensure input tensor is contiguous
    assert x.is_contiguous()
    assert x_keep.is_contiguous()
    
    # Prepare output tensor
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Calculate grid size for kernel launch
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    _dropout[grid](
        x_ptr=x,
        x_keep_ptr=x_keep,
        output_ptr=output,
        n_elements=n_elements,
        p=p,
        BLOCK_SIZE=1024,
    )
    
    return output
