import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_rowwise(
    x_ptr,
    output_ptr,
    output_maxs,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)
    
    row_start = row_idx * P2
    mask = col_idx < n_elements
    
    x = tl.load(x_ptr + row_start + col_idx, mask=mask, other=0.0)
    
    # Compute absolute values
    abs_x = tl.abs(x)
    
    # Find max value in the row
    max_val = tl.max(abs_x, axis=0)
    
    # Quantize to int8 range
    scale = 127.0 / max_val
    quantized = tl.libdevice.llrint(x * scale)
    
    # Store quantized values
    tl.store(output_ptr + row_start + col_idx, quantized, mask=mask)
    
    # Store max value for the row
    if tl.program_id(1) == 0:
        tl.store(output_maxs + row_idx, max_val)

def quantize_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_cuda, "Input tensor must be a CUDA tensor"
    assert x.dim() == 2, "Input tensor must be 2-dimensional"
    
    rows, cols = x.shape
    P2 = triton.next_power_of_2(cols)
    BLOCK_SIZE = min(P2, 1024)  # Ensure block size doesn't exceed 1024
    
    output = torch.empty_like(x, dtype=torch.int8, device='cuda')
    output_maxs = torch.empty(rows, dtype=torch.float32, device='cuda')
    
    grid = (rows,)
    
    _quantize_rowwise[grid](
        x,
        output,
        output_maxs,
        cols,
        BLOCK_SIZE=BLOCK_SIZE,
        P2=P2,
    )
    
    return output, output_maxs
