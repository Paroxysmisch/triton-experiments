import triton
import triton.language as tl
import torch

@triton.jit
def index_select_cat_bwd_kernel(grad_source, index, grad_output, n_elements, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load index and grad_output
    idx = tl.load(index + offsets, mask=mask, other=0)
    grad_out_row = tl.load(grad_output + offsets, mask=mask, other=0)

    # Update grad_source based on index
    grad_src_row = tl.load(grad_source + idx, mask=mask, other=0)
    grad_src_row += grad_out_row
    tl.store(grad_source + idx, grad_src_row, mask=mask)

def index_select_cat_bwd(grad_source, index, grad_output):
    assert grad_source.is_cuda and index.is_cuda and grad_output.is_cuda, "All tensors must be CUDA tensors"
    assert index.dtype == torch.int32, "Index tensor must be of type int32"
    assert grad_output.shape == index.shape, "Index and grad_output must have the same shape"
    
    n_elements = index.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    index_select_cat_bwd_kernel[grid](grad_source, index, grad_output, n_elements, BLOCK_SIZE=1024)
    return grad_source
