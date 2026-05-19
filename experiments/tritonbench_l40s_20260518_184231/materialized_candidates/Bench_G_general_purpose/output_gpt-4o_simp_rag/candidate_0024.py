import torch
import triton
import triton.language as tl

@triton.jit
def _dropout(
    x_ptr,
    x_keep_ptr,
    output_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Load the dropout mask
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    
    # Apply dropout: scale the kept values
    scale = 1.0 / (1.0 - p)
    output = tl.where(x_keep, x * scale, 0.0)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def dropout(x, p):
    # Prepare the output tensor
    output = torch.empty_like(x)
    assert x.is_contiguous()
    
    # Prepare the dropout mask
    x_keep = (torch.rand_like(x) > p).to(x.dtype)
    
    # Number of elements
    n_elements = x.numel()
    
    # Define the grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the Triton kernel
    _dropout[grid](
        x, x_keep, output, n_elements, p, BLOCK_SIZE=1024
    )
    
    return output

# Example usage
x = torch.randn(size=(10,)).cuda()
output = dropout(x, p=0.5)
