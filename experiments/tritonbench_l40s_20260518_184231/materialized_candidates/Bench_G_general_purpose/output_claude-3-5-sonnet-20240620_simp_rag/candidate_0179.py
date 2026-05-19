import math
import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_global(
    x_ptr,                    # pointer to input tensor
    absmax_inv_ptr,          # pointer to inverse of absolute maximum value
    output_ptr,              # pointer to output tensor
    n_elements,              # total number of elements
    BLOCK_SIZE: tl.constexpr # size of each block
):
    # Calculate block ID and offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # For the first block, compute the absolute maximum
    if pid == 0:
        abs_x = tl.abs(x)
        max_val = tl.max(tl.where(mask, abs_x, 0.0))
        absmax_inv = 127.0 / max_val
        tl.store(absmax_inv_ptr, absmax_inv)
    
    # Wait for the first block to compute absmax_inv
    tl.debug_barrier()
    
    # Load the scaling factor
    absmax_inv = tl.load(absmax_inv_ptr)
    
    # Quantize the input elements to int8
    output = tl.libdevice.llrint(x * absmax_inv)
    
    # Store the quantized output
    tl.store(output_ptr + offsets, output, mask=mask)

def quantize_global(x: torch.Tensor, block_size: int = 1024):
    """
    Globally quantize a tensor to int8 format.
    
    Args:
        x: Input tensor to quantize
        block_size: Number of elements to process per block
        
    Returns:
        tuple: (quantized tensor, absolute maximum value)
    """
    # Input validation
    assert x.is_cuda, "Input tensor must be on GPU"
    
    # Prepare output tensors
    output = torch.empty_like(x, dtype=torch.int8)
    absmax_inv = torch.empty(1, device=x.device, dtype=x.dtype)
    
    # Calculate grid size
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, block_size),)
    
    # Launch kernel
    _quantize_global[grid](
        x_ptr=x,
        absmax_inv_ptr=absmax_inv,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=block_size,
    )
    
    # Calculate the actual absmax from its inverse
    absmax = 127.0 / absmax_inv.item()
    
    return output, absmax
