import triton
import triton.language as tl
import torch

@triton.jit
def _quantize_global_kernel(
    x_ptr,                                          # Pointer to input tensor
    absmax_inv_ptr,                                 # Pointer to inverse absmax value
    output_ptr,                                     # Pointer to output tensor
    n_elements,                                     # Total number of elements
    BLOCK_SIZE: tl.constexpr                        # Number of elements per block
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute block start/end indices
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n_elements)
    
    # Create offset range for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    absmax_inv = tl.load(absmax_inv_ptr)
    
    # Quantize to int8 range (-127 to 127)
    x_scaled = x * absmax_inv * 127.0
    x_quantized = tl.math.round(x_scaled)
    
    # Clamp values to int8 range
    x_quantized = tl.minimum(127.0, tl.maximum(-127.0, x_quantized))
    
    # Store results
    tl.store(output_ptr + offsets, x_quantized, mask=mask)

def quantize_global(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Globally quantize a tensor to int8 range.
    
    Args:
        x: Input tensor to quantize
        
    Returns:
        tuple of:
            - Quantized tensor (still in fp32 format)
            - Maximum absolute value used for scaling
    """
    # Compute maximum absolute value
    absmax = torch.max(torch.abs(x)).float()
    absmax_inv = (1.0 / absmax) if absmax > 0 else 1.0
    
    # Prepare output tensor
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Define block size and compute grid
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    _quantize_global_kernel[grid](
        x_ptr=x,
        absmax_inv_ptr=absmax_inv,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output, absmax

# Example usage
if __name__ == "__main__":
    # Create sample tensor
    x = torch.randn(1000000, device='cuda')
    
    # Quantize
    x_quantized, absmax = quantize_global(x)
    
    # Verify results
    print(f"Original range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Quantized range: [{x_quantized.min():.4f}, {x_quantized.max():.4f}]")
    print(f"Absmax value: {absmax:.4f}")
