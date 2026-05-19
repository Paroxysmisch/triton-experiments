output_batch_stride,
    output_tiled_row_stride,
    output_tiled_col_stride,
    output_row_block_stride,
    output_col_block_stride,
    batch,
    M,
    N,
    nnz,
    NUM_TILES_ROW: tl.constexpr,
    NUM_TILES_COL: tl.constexpr,
    acc_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    # Kernel implementation here

@triton.jit
def _bsr_softmax_kernel(
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
    M,
    N,
    BLOCKSIZE_ROW: tl.constexpr,
    BLOCKSIZE_COL: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # Kernel implementation here

def sampled_addmm(
    alpha,
    beta,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    mat1_tiled_stride: Optional[Tuple[int, int]] = None,
    mat2_tiled_stride: Optional[Tuple[int, int]] = None,
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
    *,
    output: Optional[torch.Tensor] = None,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
):
    # Function implementation here

def bsr_softmax(values, crow_indices, col_indices, *, dtype=None):
    # Function implementation here
