import triton.language as tl
import torch

@triton.jit
def index_select_cat_bwd_kernel(
    grad_source_ptrs, grad_output_ptrs, index_values, 
    LDG_SIZE : tl.constexpr, NUM_WARPS : tl.constexpr, 
    BLOCK_SIZE_INDEX : tl.constexpr, BLOCK_SIZE_COL : tl.constexpr
):
    grid = tl.grid(BLOCK_SIZE_INDEX, BLOCK_SIZE_COL)
    row, col = tl.grid(grid.nrows, grid.ncols)
    global_id_row = row * BLOCK_SIZE_COL + col

    if  global_id_row < BLOCK_SIZE_INDEX * BLOCK_SIZE_COL:
        ldg_value = tl.load(grad_output_ptrs + global_id_row, 
                            mask=global_id_row < LDG_SIZE, 
                            other=0)
        ldg_index = index_values[global_id_row]
        accum_value = tl.shfl_down(ldg_value, 0, NUM_WARPS)
        accum_value += ldg_value if (global_id_row % NUM_WARPS) == 0 else 0

        tl.store(grad_source_ptrs + ldg_index, accum_value)

def index_select_cat_bwd(grad_source, grad_output, index):
    assert len(grad_source.shape) == 2 and len(grad_output.shape) == 2 and len(index.shape) == 1
    assert grad_source.is_cuda and grad_output.is_cuda and index.is_cuda
    assert grad_source.dtype == grad_output.dtype and grad_source.device == grad_output.device
    assert grad_source.device == index.device

    grad_source_ptrs = grad_source.data_ptr()
    grad_output_ptrs = grad_output.data_ptr()
    index_values = index.data_ptr()

    dims = (grad_source.shape[0], grad_source.shape[1])
    grid = lambda meta: (tl.ceildiv(dims[0], meta['BLOCK_SIZE_INDEX']), 
                         tl.ceildiv(dims[1], meta['BLOCK_SIZE_COL']))
    index_select_cat_bwd_kernel[grid](grad_source_ptrs, grad_output_ptrs, index_values, 
                                      LDG_SIZE=dims[0], NUM_WARPS=16, 
                                      BLOCK_SIZE_INDEX=32, BLOCK_SIZE_COL=128)
