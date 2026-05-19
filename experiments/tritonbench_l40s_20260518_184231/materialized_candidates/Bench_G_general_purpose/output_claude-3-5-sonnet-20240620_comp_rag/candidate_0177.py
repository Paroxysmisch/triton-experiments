import math
import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_rowwise(
    x_ptr,
    output_ptr,
    output_maxs_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the row index
    row_idx = tl.program_id(0)
    
    # Calculate offsets for the current row
    offsets = row_idx * n_cols + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_idx + 1) * n_cols
    
    # Load input elements for the current row
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate the absolute maximum value for normalization
    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)
    
    # Quantize the input elements to int8
    output = tl.libdevice.llrint(127.0 * (x / max_val))
    
    # Store the quantized output and max value
    tl.store(output_ptr + offsets, output, mask=mask)
    tl.store(output_maxs_ptr + row_idx, max_val)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_cols']
)
def quantize_rowwise(x: torch.Tensor):
    # Ensure input is on CUDA
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Get input dimensions
    n_rows, n_cols = x.shape
    
    # Prepare output tensors
    output = torch.empty_like(x, dtype=torch.int8)
    output_maxs = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    
    # Define grid configuration
    grid = (n_rows,)
    
    # Launch the Triton kernel
    _quantize_rowwise[grid](
        x_ptr=x,
        output_ptr=output,
        output_maxs_ptr=output_maxs,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
    )
    
    return output, output_maxs

# Example usage
if __name__ == "__main__":
    # Create a sample input tensor
    x = torch.randn(1000, 784, device='cuda')
    
    # Perform rowwise quantization
    quantized, max_values = quantize_rowwise(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Quantized output shape: {quantized.shape}")
    print(f"Max values shape: {max_values.shape}")
