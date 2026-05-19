import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_fwd_kernel(
    source, output, n_rows_source, n_cols_source,
    index, index_len,
    BLOCK_SIZE_INDEX: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr,
    OUT_PTR_BYTES: tl.constexpr
):
    # Define program ids
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    # Calculate offsets
    idx_start = pid0 * BLOCK_SIZE_INDEX
    col_start = pid1 * BLOCK_SIZE_COL
    idx_end = idx_start + BLOCK_SIZE_INDEX
    col_end = col_start + BLOCK_SIZE_COL

    # Create offset grids
    index_grid_x = tl.arange(0, n_rows_source)
    index_grid_y = index.repeat(n_rows_source)
    col_grid_x = tl.arange(0, n_rows_source)
    col_grid_y = tl.arange(0, n_cols_source)

    # Apply masks
    index_grid_mask = (idx_start <= index_grid_x) & (index_grid_x < idx_end)
    col_grid_mask = (col_start <= col_grid_y) & (col_grid_y < col_end)

    # Calculate outputs
    source_ptr = source + index_grid_x[:, None] * n_cols_source + index_grid_y[None, :]
    output_ptr = output + idx_start * n_cols_source + col_start + OUT_PTR_BYTES
    index_grid = tl.load(source_ptr, mask = index_grid_mask)
    tl.store(output_ptr, index_grid, mask=col_grid_mask)


def index_select_cat_fwd(source, index):
    assert source.is_cuda and index.is_cuda, "Both source and index must be on GPU"
    assert source.ndim == 2 and index.ndim == 1, "Source must be 2D and index must be 1D"
    assert index.numel() <= source.size(0), "Index length exceed number of rows in the source"

    n_rows_source, n_cols_source = source.shape
    out_shape = (index.numel(), n_cols_source)
    output = torch.empty(out_shape, dtype=source.dtype, device=source.device)

    # Get strides
    index_stride = index.stride(0)
    source_strides = source.stride()
    output_stride = output.stride(0)

    # Define constants and grid
    cuda_dtype = source.dtype if source.dtype.itemsize <= 4 else torch.int32
    n_elements_source = n_rows_source * n_cols_source
    n_elements_out = index.numel() * n_cols_source
    n_elements_stride = n_elements_source * source_strides[1]
    BLOCK_SIZE_INDEX = triton.next_power_of_2(n_rows_source)
    BLOCK_SIZE_COL = triton.next_power_of_2(n_cols_source)
    OUT_PTR_BYTES = tl.ptrdiff(output.data_ptr())
    N_ITER = triton.cdiv(n_elements_out, BLOCK_SIZE_INDEX * BLOCK_SIZE_COL)

    # Call the kernel with gris
    index_select_cat_fwd_kernel[N_ITER, BLOCK_SIZE_INDEX, BLOCK_SIZE_COL](
        source.data_ptr(),
        output.data_ptr(),
        n_rows_source, n_cols_source,
        index.data_ptr(), index.numel(),
        BLOCK_SIZE_INDEX, BLOCK_SIZE_COL,
        OUT_PTR_BYTES
    )
    return output.view(-1, n_cols_source)
