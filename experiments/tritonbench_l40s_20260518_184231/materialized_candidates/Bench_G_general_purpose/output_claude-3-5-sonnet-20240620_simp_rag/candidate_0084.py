import triton
import triton.language as tl
import torch

@triton.jit
def index_select_cat_bwd_kernel(
    grad_source_ptr,
    index_ptr,
    grad_output_ptr,
    n_elements,
    source_stride_0,
    source_stride_1,
    index_stride_0,
    output_stride_0,
    output_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for parallel execution
    pid = tl.program_id(0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Calculate row and col indices
    row_idx = offsets // output_stride_0
    col_idx = offsets % output_stride_1
    
    # Load index values for this block
    index = tl.load(index_ptr + row_idx * index_stride_0, mask=mask)
    
    # Load grad_output values
    grad_vals = tl.load(
        grad_output_ptr + row_idx * output_stride_0 + col_idx,
        mask=mask
    )
    
    # Calculate source indices
    source_offset = index * source_stride_0 + col_idx
    
    # Atomic add to handle index collisions
    tl.atomic_add(
        grad_source_ptr + source_offset,
        grad_vals,
        mask=mask
    )

def index_select_cat_bwd(grad_source, index, grad_output):
    """
    Backward pass for concatenated index select operation.
    
    Args:
        grad_source: Gradient tensor for source (2D CUDA tensor)
        index: Index tensor (1D CUDA tensor)
        grad_output: Gradient tensor from output (2D CUDA tensor)
    """
    # Input validation
    assert grad_source.is_cuda and index.is_cuda and grad_output.is_cuda
    assert grad_source.dim() == 2 and grad_output.dim() == 2
    assert index.dim() == 1
    assert grad_source.dtype == grad_output.dtype
    
    # Get dimensions
    n_elements = grad_output.numel()
    
    # Get strides
    source_stride_0 = grad_source.stride(0)
    source_stride_1 = grad_source.stride(1)
    index_stride_0 = index.stride(0)
    output_stride_0 = grad_output.stride(0)
    output_stride_1 = grad_output.stride(1)
    
    # Configure block size and grid
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    index_select_cat_bwd_kernel[grid](
        grad_source.data_ptr(),
        index.data_ptr(),
        grad_output.data_ptr(),
        n_elements,
        source_stride_0,
        source_stride_1,
        index_stride_0,
        output_stride_0,
        output_stride_1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return grad_source
