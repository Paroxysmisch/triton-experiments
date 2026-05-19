import math
import torch
import triton
import triton.language as tl

# Triton kernel for global quantization
@triton.jit
def _quantize_global(
    x_ptr,
    absmax_inv_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the block index and the element offsets within the block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + arange
    mask = offsets < n_elements
    
    # Load the input elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate the absolute maximum value for normalization
    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)
    
    # Store the inverse of the maximum absolute value
    absmax_inv = 1.0 / max_val
    tl.store(absmax_inv_ptr, absmax_inv)
    
    # Quantize the input elements to int8
    output = tl.libdevice.llrint(127.0 * (x * absmax_inv))
    
    # Store the quantized output
    tl.store(output_ptr + offsets, output, mask=mask)

def quantize_global(x: torch.Tensor):
    # Prepare output tensor
    output = torch.empty_like(x, dtype=torch.int8)
    absmax_inv = torch.empty(1, device=x.device, dtype=torch.float32)

    # Ensure CUDA compatibility
    assert x.is_cuda and output.is_cuda

    # Define grid configuration
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Example block size
    grid = lambda meta: (math.ceil(n_elements / BLOCK_SIZE),)
    
    # Launch the Triton kernel
    _quantize_global[grid](x, absmax_inv, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute the maximum absolute value
    absmax = 1.0 / absmax_inv.item()
    
    return output, absmax
