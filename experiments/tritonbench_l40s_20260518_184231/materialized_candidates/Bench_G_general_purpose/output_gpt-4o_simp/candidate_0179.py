import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def _quantize_global(x_ptr, absmax_inv_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define the block index
    pid = tl.program_id(axis=0)
    
    # Calculate the start position for this block
    block_start = pid * BLOCK_SIZE
    
    # Load the input data
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load the inverse of the maximum absolute value
    absmax_inv = tl.load(absmax_inv_ptr)
    
    # Quantize the data
    quantized = x * absmax_inv * 127.0
    quantized = tl.clamp(quantized, -127.0, 127.0)
    
    # Convert to int8
    quantized_int8 = quantized.to(tl.int8)
    
    # Store the result
    tl.store(output_ptr + offsets, quantized_int8, mask=mask)

# Define the wrapper function
def quantize_global(x, BLOCK_SIZE=1024):
    # Ensure input is a CUDA tensor
    assert x.is_cuda, "Input tensor must be on CUDA"
    
    # Compute the maximum absolute value
    absmax = torch.max(torch.abs(x))
    
    # Compute the inverse of the maximum absolute value
    absmax_inv = 1.0 / absmax if absmax != 0 else 0.0
    
    # Prepare output tensor
    output = torch.empty_like(x, dtype=torch.int8)
    
    # Launch the Triton kernel
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    _quantize_global[grid](x, absmax_inv, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output, absmax

# Example usage
x = torch.randn(10000, device='cuda')
quantized_output, absmax = quantize_global(x)
print(quantized_output)
print(absmax)
