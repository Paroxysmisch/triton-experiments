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
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the input elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load the precomputed inverse of the maximum absolute value
    absmax_inv = tl.load(absmax_inv_ptr)
    
    # Quantize the input elements to int8
    output = tl.libdevice.llrint(127.0 * x * absmax_inv)
    
    # Store the quantized output
    tl.store(output_ptr + offsets, output.to(tl.int8), mask=mask)

def quantize_global(x: torch.Tensor):
    # Compute the maximum absolute value and its inverse
    absmax = torch.max(torch.abs(x))
    absmax_inv = 1.0 / absmax
    
    # Prepare output tensor
    output = torch.empty_like(x, dtype=torch.int8)
    
    # Ensure CUDA compatibility
    assert x.is_cuda and output.is_cuda
    
    # Number of elements in the input tensor
    n_elements = x.numel()
    
    # Define block size and grid size
    BLOCK_SIZE = 1024  # You can tune this value for different workloads
    grid = lambda meta: (math.ceil(n_elements / BLOCK_SIZE),)
    
    # Launch the Triton kernel
    _quantize_global[grid](x, absmax_inv, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output, absmax

# Example usage
if __name__ == "__main__":
    x = torch.randn(10000, device='cuda')
    output, absmax = quantize_global(x)
    print(output)
    print(absmax)
