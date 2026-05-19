import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_bwd_kernel(
    grad_output_ptr,
    grad_output_row_stride,
    grad_output_col_stride,
    indices_ptr,
    indices_stride,
    grad_source_ptr,
    grad_source_row_stride,
    grad_source_col_stride,
    grad_output_num_rows,
    grad_output_num_cols,
    grad_source_num_rows,
    BLOCK_SIZE_INDEX: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid_index = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    row_block = pid_index * BLOCK_SIZE_INDEX
    row_offsets = row_block + tl.arange(0, BLOCK_SIZE_INDEX)
    row_mask = row_offsets < grad_output_num_rows
    
    col_block = pid_col * BLOCK_SIZE_COL
    col_offsets = col_block + tl.arange(0, BLOCK_SIZE_COL)
    col_mask = col_offsets < grad_output_num_cols
    
    for i in tl.range(BLOCK_SIZE_INDEX):
        row = row_block + i
        if row >= grad_output_num_rows:
            continue
        
        k = tl.load(indices_ptr + row * indices_stride)
        if k >= grad_source_num_rows:
            continue
        
        grad_output_offsets = row * grad_output_row_stride + col_offsets * grad_output_col_stride
        grad_source_offsets = k * grad_source_row_stride + col_offsets * grad_source_col_stride
        
        grads = tl.load(grad_output_ptr + grad_output_offsets, mask=col_mask, other=0)
        tl.atomic_add(grad_source_ptr + grad_source_offsets, grads, mask=col_mask)

def index_select_cat_bwd(grad_output: torch.Tensor, indices: torch.Tensor, grad_source: torch.Tensor) -> torch.Tensor:
    assert grad_output.is_cuda and indices.is_cuda and grad_source.is_cuda
    assert grad_output.dim() == 2, "grad_output must be 2D"
    assert indices.dim() == 1, "indices must be 1D"
    assert grad_source.dim() == 2, "grad_source must be 2D"
    assert indices.size(0) == grad_output.size(0), "Indices and grad_output must have the same number of rows"
    assert grad_output.size(1) == grad_source.size(1), "grad_output and grad_source must have the same number of columns"
    assert grad_output.is_contiguous(), "grad_output must be contiguous"
    assert indices.is_contiguous(), "indices must be contiguous"
    assert grad_source.is_contiguous(), "grad_source must be contiguous"
    assert (indices < grad_source.size(0)).all(), "Indices exceed grad_source dimensions"
    
    BLOCK_SIZE_INDEX = 128
    BLOCK_SIZE_COL = 64
    grid = (
        (grad_output.size(0) + BLOCK_SIZE_INDEX - 1) // BLOCK_SIZE_INDEX,
        (grad_output.size(1) + BLOCK_SIZE_COL - 1) // BLOCK_SIZE_COL,
    )
    
    index_select_cat_bwd_kernel[grid](
        grad_output_ptr=grad_output.data_ptr(),
        grad_output_row_stride=grad_output.stride(0),
        grad_output_col_stride=grad_output.stride(1),
        indices_ptr=indices.data_ptr(),
        indices_stride=indices.stride(0),
        grad_source_ptr=grad_source.data_ptr(),
        grad_source_row_stride=grad_source.stride(0),
        grad_source_col_stride=grad_source.stride(1),
        grad_output_num_rows=grad_output.size(0),
        grad_output_num_cols=grad_output.size(1),
        grad_source_num_rows=grad_source.size(0),
        BLOCK_SIZE_INDEX=BLOCK_SIZE_INDEX,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )
    
    return grad_source
