import triton
import triton.language as tl
import torch

BLOCK_SIZE_INDEX = 128
BLOCK_SIZE_COL = 128

@triton.jit
def index_select_cat_bwd_kernel(
    grad_output_ptr, 
    grad_source_ptr, 
    index_ptr, 
    num_rows, 
    num_cols,
    grad_output_stride0, 
    grad_output_stride1,
    grad_source_stride0, 
    grad_source_stride1,
    index_stride0,
    BLOCK_SIZE_INDEX: tl.constexpr, 
    BLOCK_SIZE_COL: tl.constexpr
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_block_start = pid_row * BLOCK_SIZE_INDEX
    col_block_start = pid_col * BLOCK_SIZE_COL

    row_offsets = row_block_start + tl.arange(0, BLOCK_SIZE_INDEX)
    col_offsets = col_block_start + tl.arange(0, BLOCK_SIZE_COL)

    row_mask = row_offsets < num_rows
    col_mask = col_offsets < num_cols
    mask = row_mask[:, None] & col_mask[None, :]

    idx = tl.load(index_ptr + row_offsets * index_stride0, mask=row_mask, other=0)
    idx = idx[:, None]

    grad_output_offset = (row_offsets[:, None] * grad_output_stride0) + (col_offsets[None, :] * grad_output_stride1)
    go = tl.load(grad_output_ptr + grad_output_offset, mask=mask, other=0.0)

    grad_source_offset = (idx * grad_source_stride0) + (col_offsets[None, :] * grad_source_stride1)
    tl.atomic_add(grad_source_ptr + grad_source_offset, go, mask=mask)

def index_select_cat_bwd(grad_output: torch.Tensor, index: torch.Tensor, grad_source: torch.Tensor) -> torch.Tensor:
    assert grad_output.is_cuda
    assert index.is_cuda
    assert grad_source.is_cuda
    assert grad_output.dim() == 2
    assert index.dim() == 1
    assert grad_source.dim() == 2
    assert grad_output.shape[0] == index.shape[0]

    num_rows = index.shape[0]
    num_cols = grad_output.shape[1]

    grid = (
        (num_rows + BLOCK_SIZE_INDEX - 1) // BLOCK_SIZE_INDEX, 
        (num_cols + BLOCK_SIZE_COL - 1) // BLOCK_SIZE_COL
    )

    index_select_cat_bwd_kernel[grid](
        grad_output, 
        grad_source, 
        index,
        num_rows, 
        num_cols,
        grad_output.stride(0), 
        grad_output.stride(1),
        grad_source.stride(0), 
        grad_source.stride(1),
        index.stride(0),
        BLOCK_SIZE_INDEX=BLOCK_SIZE_INDEX, 
        BLOCK_SIZE_COL=BLOCK_SIZE_COL
    )
    return grad_source
