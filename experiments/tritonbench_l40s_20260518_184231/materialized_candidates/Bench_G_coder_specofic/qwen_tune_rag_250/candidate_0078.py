crow_indices_stride,
    col_indices_ptr,
    col_indices_batch_stride,
    col_indices_stride,
    dense_ptr,
    dense_batch_stride,
    dense_tiled_row_stride,
    dense_tiled_col_stride,
    dense_row_block_stride,
    dense_col_block_stride,
    output_ptr,
    output_batch_stride,
    output_tiled_row_stride,
    output_tiled_col_stride,
    output_row_block_stride,
    output_col_block_stride,
    nnz,
    row_block_size,
    col_block_size,
    TILE_ROW: tl.constexpr,
    TILE_COL: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    # Kernel implementation here

@triton.jit
def _bsr_softmax_kernel(
    BLOCKSIZE_ROW: tl.constexpr,
    BLOCKSIZE_COL: tl.constexpr,
    values_ptr,
    values_batch_stride,
    values_nnz_stride,
    values_row_block_stride,
    values_col_block_stride,
    crow_indices_ptr,
    crow_indices_batch_stride,
    crow_indices_stride,
    col_indices_ptr,
    col_indices_batch_stride,
    col_indices_stride,
    max_rows_per_program,
    nnz,
    row_block_size,
    col_block_size,
    TILE_ROW: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    # Kernel implementation here

def sampled_addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta=1.0,
    alpha=1.0,
    out: Optional[torch.Tensor] = None,
    skip_checks: bool = False,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
):
    # Function implementation here

def bsr_strided_dense_rowspace(
    values: torch.Tensor,
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    dense: torch.Tensor,
    sparsity_block_size: int,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
) -> torch.Tensor:
    # Function implementation here

def bsr_softmax(
    values: torch.Tensor,
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    sparsity_block_size: int,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
) -> torch.Tensor:
    # Function implementation here
