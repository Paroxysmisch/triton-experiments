import triton
import triton.language as tl
import torch

@triton.jit
def index_select_cat_bwd_kernel(
    grad_output_ptr, index_ptr, grad_source_ptr,
    n_indices, n_cols, BLOCK_SIZE_INDEX: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr
):
    pid = tl.program_id(0)
    block_start_index = pid * BLOCK_SIZE_INDEX
    offsets_index = block_start_index + tl.arange(0, BLOCK_SIZE_INDEX)
    mask_index = offsets_index < n_indices

    for col in range(0, n_cols, BLOCK_SIZE_COL):
        offsets_col = col + tl.arange(0, BLOCK_SIZE_COL)
        mask_col = offsets_col < n_cols

        index_vals = tl.load(index_ptr + offsets_index, mask=mask_index, other=0)
        grad_output_vals = tl.load(grad_output_ptr + index_vals[:, None] * n_cols + offsets_col[None, :], mask=mask_index[:, None] & mask_col[None, :], other=0)

        tl.atomic_add(grad_source_ptr + offsets_index[:, None] * n_cols + offsets_col[None, :], grad_output_vals, mask=mask_index[:, None] & mask_col[None, :])

def index_select_cat_bwd(grad_output, index, grad_source):
    # Ensure inputs are 2D and on the same CUDA device
    assert grad_output.dim() == 2, "grad_output must be a 2D tensor"
    assert index.dim() == 1, "index must be a 1D tensor"
    assert grad_source.dim() == 2, "grad_source must be a 2D tensor"
    assert grad_output.is_cuda, "grad_output must be a CUDA tensor"
    assert index.is_cuda, "index must be a CUDA tensor"
    assert grad_source.is_cuda, "grad_source must be a CUDA tensor"
    assert grad_output.dtype == grad_source.dtype, "grad_output and grad_source must have the same dtype"
    assert grad_output.shape[1] == grad_source.shape[1], "grad_output and grad_source must have the same number of columns"

    n_indices, n_cols = grad_output.shape
    grad_source.zero_()  # Zero out the grad_source before accumulation

    grid = lambda meta: (triton.cdiv(n_indices, meta['BLOCK_SIZE_INDEX']),)
    index_select_cat_bwd_kernel[grid](grad_output, index, grad_source, n_indices, n_cols, BLOCK_SIZE_INDEX=128, BLOCK_SIZE_COL=128)

    return grad_source
