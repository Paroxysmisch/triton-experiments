import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_global(
    x_ptr,
    absmax_inv_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program ID and offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input values and absmax_inv
    x = tl.load(x_ptr + offsets, mask=mask)
    absmax_inv = tl.load(absmax_inv_ptr)
    
    # Quantize to int8 range (-127 to 127)
    x_scaled = x * absmax_inv * 127.0
    x_quantized = tl.extra.cuda.libdevice.llrint(x_scaled)
    
    # Store results
    tl.store(output_ptr + offsets, x_quantized, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_elements'],
)
def quantize_global(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Input validation
    assert x.is_cuda and x.is_contiguous()
    n_elements = x.numel()
    
    # Calculate absmax and its inverse
    absmax = torch.max(torch.abs(x)).clone()
    absmax_inv = (1.0 / absmax).clone()
    
    # Prepare output tensor
    output = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    _quantize_global[grid](
        x_ptr=x,
        absmax_inv_ptr=absmax_inv,
        output_ptr=output,
        n_elements=n_elements,
    )
    
    return output, absmax

# Example usage
def main():
    # Create sample input tensor
    x = torch.randn(1000000, device='cuda')
    
    # Quantize
    x_quantized, absmax = quantize_global(x)
    
    # Verify results
    x_dequantized = (x_quantized.float() / 127.0) * absmax
    max_error = torch.max(torch.abs(x - x_dequantized))
    print(f"Maximum quantization error: {max_error:.6f}")

if __name__ == "__main__":
    main()
