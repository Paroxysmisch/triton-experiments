import triton
import triton.language as tl
import torch

@triton.jit
def _dropout(
    x_ptr,          # pointer to input tensor
    output_ptr,     # pointer to output tensor
    x_keep_ptr,     # pointer to dropout mask
    n_elements,     # number of elements in the tensor
    p,              # dropout probability
    BLOCK_SIZE: tl.constexpr,  # number of elements per block
):
    # Calculate the program ID and the block start
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create an offset array for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements (handling the last block)
    mask = offsets < n_elements
    
    # Load input data and dropout mask using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    keep = tl.load(x_keep_ptr + offsets, mask=mask)
    
    # Apply dropout: multiply by keep mask and scale by 1/(1-p)
    scale = 1.0 / (1.0 - p)
    output = tl.where(keep, x * scale, 0.0)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def dropout(x: torch.Tensor, p: float) -> torch.Tensor:
    """
    Applies dropout to input tensor.
    
    Args:
        x: Input tensor
        p: Dropout probability
    
    Returns:
        Output tensor with dropout applied
    """
    # Input validation
    if not 0.0 <= p < 1.0:
        raise ValueError("Dropout probability must be in [0, 1)")
    
    # If p is 0, return input unchanged
    if p == 0.0:
        return x
    
    # Initialize output tensor and generate dropout mask
    output = torch.empty_like(x)
    keep = torch.bernoulli(torch.full_like(x, 1 - p))
    
    # Calculate grid size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    
    # Launch kernel
    _dropout[grid](
        x_ptr=x,
        output_ptr=output,
        x_keep_ptr=keep,
        n_elements=x.numel(),
        p=p,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
