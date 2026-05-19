import triton
import triton.language as tl
import torch

@triton.jit
def index_select_cat_bwd_kernel(
    grad_output_ptr, source_ptr, grad_source_ptr, index_ptr,
    source_grad_rows, source_grad_cols, index_size,
    BLOCK_SIZE_INDEX: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr,
):
    pid_index = tl.program_id(0)
    pid_col = tl.program_id(1)

    offset_index = pid_index * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    mask_index = offset_index < index_size
    offset_col = pid_col * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    mask_col = offset_col < source_grad_cols
    mask = mask_index & mask_col

    offset_index_2 = pid_index * BLOCK_SIZE_INDEX
    offset_col_2 = pid_col
    index = tl.load(index_ptr + offset_index, mask=mask_index)
    grad_output_ptrs = grad_output_ptr + (index + offset_index_2) * source_grad_cols + offset_col_2
    source_ptrs = source_ptr + offset_index * source_grad_cols + offset_col
    grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0)
    source = tl.load(source_ptrs, mask=mask, other=0.0)

    source_grad = grad_output * source
    tl.store(grad_source_ptr + offset_index * source_grad_cols + offset_col, source_grad, mask=mask)

def index_select_cat_bwd(
    grad_output: torch.Tensor, source: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    assert grad_output.ndim == 2 and grad_output.is_contiguous()
    assert source.ndim == 2 and source.is_contiguous()
    assert index.ndim == 1 and index.is_contiguous()
    assert grad_output.stride(0) == 1
    assert source.stride(0) == 1
    assert index.stride(0) == 1
    assert index.size(0) > 0
    assert grad_output.size(0) % index.size(0) == 0

    grad_source = torch.empty_like(source, dtype=grad_output.dtype, device=grad_output.device)
    index_size = index.size(0)
    source_grad_rows = grad_output.size(0) // index_size
    source_grad_cols = source.size(1)

    grid = lambda meta: (
        triton.cdiv(index_size, meta["BLOCK_SIZE_INDEX"]),
        triton.cdiv(source_grad_cols, meta["BLOCK_SIZE_COL"]),
    )
    index_select_cat_bwd_kernel[grid](
        grad_output, source, grad_source, index,
        source_grad_rows, source_grad_cols, index_size,
        BLOCK_SIZE_INDEX=128, BLOCK_SIZE_COL=32,
    )
    return grad_source
