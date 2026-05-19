import triton
import triton.language as tl
import torch

@triton.jit
def _dropout(
    x_ptr,
    x_keep_ptr,
    output_ptr,
    p,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load data from input tensor and mask
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask, other=0)
    
    # Compute scaling factor and apply dropout
    scale = 1.0 / (1 - p)
    output = tl.where(x_keep != 0, x * scale, 0.0)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def dropout(x: torch.Tensor, x_keep: torch.Tensor, p: float):
    # Ensure inputs are contiguous and valid
    assert x.is_contiguous(), "Input tensor x must be contiguous"
    assert x_keep.is_contiguous(), "Mask tensor x_keep must be contiguous"
    assert x.shape == x_keep.shape, "Input and mask must have the same shape"
    assert 0 <= p < 1, "Dropout probability must be in [0, 1)"
    
    output = torch.empty_like(x)
    N = x.numel()
    
    # Compute grid size and launch kernel
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    _dropout[grid](x, x_keep, output, p, N, BLOCK_SIZE=1024)
    
    return output
