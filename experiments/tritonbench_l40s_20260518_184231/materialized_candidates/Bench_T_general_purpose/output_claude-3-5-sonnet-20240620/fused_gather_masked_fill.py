import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def _fused_gather_masked_fill_kernel(
    # Pointers to tensors
    output_ptr, input_ptr, index_ptr, mask_ptr,
    # Shapes and strides
    n_elements, dim, value,
    # Strides for input/output/index/mask tensors
    output_stride, input_stride, index_stride, mask_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    # Compute block start/end
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n_elements)
    
    # Compute offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices
    indices = tl.load(index_ptr + offsets * index_stride, mask=mask)
    
    # Load input values using gathered indices
    input_offsets = indices * input_stride[dim]
    input_vals = tl.load(input_ptr + input_offsets, mask=mask)
    
    # Load mask values
    mask_vals = tl.load(mask_ptr + offsets * mask_stride, mask=mask)
    
    # Apply masked fill
    output = tl.where(mask_vals, value, input_vals)
    
    # Store result
    tl.store(output_ptr + offsets * output_stride, output, mask=mask)

def fused_gather_masked_fill(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    mask: torch.Tensor,
    value: float,
    *,
    sparse_grad: bool = False,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # Input validation
    assert input.dim() == index.dim(), "Input and index must have same number of dimensions"
    assert mask.is_floating_point() or mask.dtype == torch.bool, "Mask must be boolean tensor"
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Prepare shapes and strides
    n_elements = index.numel()
    
    # Get strides
    output_stride = out.stride()
    input_stride = input.stride()
    index_stride = index.stride()
    mask_stride = mask.stride()
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    _fused_gather_masked_fill_kernel[grid](
        out.data_ptr(),
        input.data_ptr(),
        index.data_ptr(),
        mask.data_ptr(),
        n_elements,
        dim,
        value,
        output_stride[0],
        input_stride,
        index_stride[0],
        mask_stride[0],
        BLOCK_SIZE=1024,
    )
    
    return out
