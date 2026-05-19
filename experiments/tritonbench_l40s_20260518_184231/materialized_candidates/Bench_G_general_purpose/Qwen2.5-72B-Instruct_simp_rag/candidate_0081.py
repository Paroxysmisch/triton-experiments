import triton
import triton.language as tl
import torch

@triton.jit
def index_select_cat_bwd_kernel(grad_source, index, grad_output, n_elements, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the index and grad_output
    index_row = tl.load(index + offsets, mask=mask, other=0)
    grad_output_row = tl.load(grad_output + offsets, mask=mask, other=0)

    # Add the selected gradients to grad_source
    tl.atomic_add(grad_source + index_row, grad_output_row, mask=mask)

def index_select_cat_bwd(grad_source, index, grad_output):
    # Validate input shapes and strides
    assert grad_source.dim() == 2, "grad_source must be a 2D tensor"
    assert index.dim() == 1, "index must be a 1D tensor"
    assert grad_output.dim() == 2, "grad_output must be a 2D tensor"
    assert grad_source.size(0) == grad_output.size(0), "grad_source and grad_output must have the same number of rows"
    assert index.size(0) == grad_output.size(1), "index and grad_output must have the same number of columns"

    # Get the number of elements to process
    n_elements = grad_output.numel()

    # Define the grid and block size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Invoke the Triton kernel
    index_select_cat_bwd_kernel[grid](grad_source, index, grad_output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return grad_source
