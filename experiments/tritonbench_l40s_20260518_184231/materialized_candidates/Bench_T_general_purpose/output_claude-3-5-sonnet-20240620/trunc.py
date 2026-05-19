import triton
import triton.language as tl
from typing import Optional

@triton.jit
def _trunc_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements in tensor
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute trunc - for floating point numbers
    # For integers, this will effectively be a copy
    result = tl.where(x >= 0, tl.floor(x), tl.ceil(x))
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

def trunc(input, *, out: Optional[Tensor] = None) -> Tensor:
    """
    Returns a new tensor with the truncated integer values of the elements of the input tensor.
    
    Args:
        input (Tensor): the input tensor
    
    Keyword args:
        out (Tensor, optional): the output tensor
        
    Returns:
        Tensor: A tensor containing the truncated values of input
    """
    if out is None:
        out = input.new_empty(input.shape)
    
    # If input is integer type, just make a copy
    if input.dtype.is_integer:
        return input.clone() if out is None else out.copy_(input)
    
    # Calculate grid and block size
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    _trunc_kernel[grid, BLOCK_SIZE](
        input.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE
    )
    
    return out
