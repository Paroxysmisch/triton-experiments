import math
import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_rowwise(
    x_ptr,          # Pointer to input tensor [M, N]
    output_ptr,     # Pointer to output tensor [M, N]
    output_maxs,    # Pointer to max values tensor [M]
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements per row
    P2: tl.constexpr,          # Next power of 2 >= BLOCK_SIZE
):
    # Get the row index (program ID corresponds to row number)
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this row
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for elements in this row
    offsets = block_start + tl.arange(0, P2)
    
    # Mask for valid elements (in case P2 > BLOCK_SIZE)
    row_mask = tl.arange(0, P2) < BLOCK_SIZE
    
    # Load elements for this row
    x = tl.load(x_ptr + offsets, mask=row_mask)
    
    # Find maximum absolute value in the row
    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(row_mask, abs_x, 0.0), axis=0)
    
    # Avoid division by zero
    max_val = tl.where(max_val == 0.0, 1.0, max_val)
    
    # Scale to int8 range [-127, 127] and round to nearest integer
    scaled = tl.libdevice.llrint(127.0 * (x / max_val))
    
    # Store quantized values and max value
    tl.store(output_ptr + offsets, scaled, mask=row_mask)
    tl.store(output_maxs + pid, max_val)

def quantize_rowwise(x: torch.Tensor):
    """
    Quantize input tensor row-wise to int8.
    
    Args:
        x: Input tensor of shape [M, N] on GPU
    
    Returns:
        tuple: (quantized tensor, row-wise max values)
    """
    assert x.dim() == 2, "Input tensor must be 2-dimensional"
    assert x.is_cuda, "Input tensor must be on GPU"
    
    # Prepare output tensors
    output = torch.empty_like(x, dtype=torch.int8)
    output_maxs = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    
    # Calculate next power of 2 for block size
    P2 = 2 ** math.ceil(math.log2(x.shape[1]))
    
    # Launch kernel with one thread block per row
    grid = (x.shape[0],)
    _quantize_rowwise[grid](
        x_ptr=x,
        output_ptr=output,
        output_maxs=output_maxs,
        n_elements=x.numel(),
        BLOCK_SIZE=x.shape[1],
        P2=P2,
    )
    
    return output, output_maxs
