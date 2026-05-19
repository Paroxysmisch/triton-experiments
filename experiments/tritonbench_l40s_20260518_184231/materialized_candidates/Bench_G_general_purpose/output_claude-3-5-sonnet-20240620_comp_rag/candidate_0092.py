import math
import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_rowwise(
    x_ptr,                    # Pointer to input tensor [M, N]
    output_ptr,              # Pointer to output tensor [M, N]
    output_maxs_ptr,         # Pointer to max values [M]
    n_elements,              # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Size of each row (N)
    P2: tl.constexpr,        # Power of 2 ceiling of BLOCK_SIZE
):
    # Get the row index (program ID corresponds to row number)
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this row
    row_start = pid * BLOCK_SIZE
    
    # Create offset array for the row elements
    offsets = row_start + tl.arange(0, P2)
    
    # Create mask for valid elements in the row
    row_mask = tl.arange(0, P2) < BLOCK_SIZE
    
    # Load input elements for this row
    x = tl.load(x_ptr + offsets, mask=row_mask, other=0.0)
    
    # Compute absolute values and find maximum
    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(row_mask, abs_x, float('-inf')))
    
    # Avoid division by zero
    max_val = tl.where(max_val == 0, 1.0, max_val)
    
    # Scale to int8 range [-127, 127] and round to nearest integer
    scaled_x = tl.libdevice.llrint(127.0 * (x / max_val))
    
    # Store quantized values and max value
    tl.store(output_ptr + offsets, scaled_x, mask=row_mask)
    tl.store(output_maxs_ptr + pid, max_val)

def quantize_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor row-wise to int8.
    
    Args:
        x: Input tensor of shape [M, N] on CUDA device
        
    Returns:
        tuple of:
        - Quantized tensor of shape [M, N] with dtype int8
        - Max absolute values per row of shape [M] with dtype float32
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert len(x.shape) == 2, "Input tensor must be 2D"
    
    # Prepare output tensors
    M, N = x.shape
    output = torch.empty_like(x, dtype=torch.int8)
    output_maxs = torch.empty(M, dtype=torch.float32, device=x.device)
    
    # Calculate power of 2 ceiling for the row size
    P2 = 2 ** math.ceil(math.log2(N))
    
    # Launch kernel with grid size = number of rows
    grid = (M,)
    _quantize_rowwise[grid](
        x_ptr=x,
        output_ptr=output,
        output_maxs_ptr=output_maxs,
        n_elements=x.numel(),
        BLOCK_SIZE=N,
        P2=P2,
    )
    
    return output, output_maxs
