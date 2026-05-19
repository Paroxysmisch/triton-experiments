import math
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
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    global_mask = offsets < n_elements
    mask = row_mask & global_mask
    
    # Load input elements with boundary checks
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute absolute values and masked max
    abs_x = tl.abs(x)
    valid_abs_x = tl.where(mask, abs_x, -tl.math.INFINITY)
    max_val = tl.max(valid_abs_x, axis=0)
    max_val = tl.maximum(max_val, 1e-12)  # Prevent division by zero
    
    # Quantize and store
    quantized = tl.libdevice.llrint(x * (127.0 / max_val))
    tl.store(output_ptr + offsets, quantized, mask=mask)
    tl.store(output_maxs + pid, max_val)

def quantize_rowwise(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.dim() == 2, "Input tensor must be 2-dimensional"
    
    rows, cols = x.shape
    P2 = 1 << (cols - 1).bit_length()  # Next power of two
    
    # Prepare output tensors
    output = torch.empty_like(x, dtype=torch.int8)
    output_maxs = torch.empty(rows, device=x.device, dtype=torch.float16)
    
    # Launch kernel with appropriate configuration
    grid = lambda meta: (rows,)
    _quantize_rowwise[grid](
        x_ptr=x, 
        output_ptr=output,
        output_maxs=output_maxs,
        n_elements=output.numel(),
        BLOCK_SIZE=cols,
        P2=P2
    )
    return output, output_maxs
