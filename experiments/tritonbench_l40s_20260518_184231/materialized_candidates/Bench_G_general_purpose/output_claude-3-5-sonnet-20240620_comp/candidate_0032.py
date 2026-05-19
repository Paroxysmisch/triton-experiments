import triton
import triton.language as tl
import torch

@triton.jit
def _dropout(
    x_ptr,
    x_keep_ptr,
    output_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(axis=0)
    
    # Compute the block start and end
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n_elements)
    
    # Create a mask for the valid elements in the block
    mask = tl.arange(0, BLOCK_SIZE) < (block_end - block_start)
    
    # Load the input and keep mask
    x = tl.load(x_ptr + block_start, mask=mask)
    x_keep = tl.load(x_keep_ptr + block_start, mask=mask)
    
    # Apply dropout
    output = tl.where(x_keep, x / (1 - p), 0.0)
    
    # Store the result
    tl.store(output_ptr + block_start, output, mask=mask)

def dropout(x: torch.Tensor, p: float) -> torch.Tensor:
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Generate the keep mask
    x_keep = torch.rand_like(x) > p
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Calculate grid size
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    
    # Launch kernel
    _dropout[grid](
        x.data_ptr(),
        x_keep.data_ptr(),
        output.data_ptr(),
        n_elements,
        p,
        BLOCK_SIZE=1024,
    )
    
    return output
