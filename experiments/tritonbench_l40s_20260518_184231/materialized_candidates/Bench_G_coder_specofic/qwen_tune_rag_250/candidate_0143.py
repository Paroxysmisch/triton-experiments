_stride,
    dense_tiled_col_stride,
    dense_row_block_stride,
    dense_col_block_stride,
    output_ptr,
    output_batch_stride,
    output_tiled_row_stride,
    output_tiled_col_stride,
    output_row_block_stride,
    output_col_block_stride,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
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
    BLOCKSIZE_ROW: tl.constexpr,
    BLOCKSIZE_COL: tl.constexpr,
    output_ptr,
    output_batch_stride,
    output_nnz_stride,
    output_row_block_stride,
    output_col_block_stride,
    allow_tf32: tl.constexpr,
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
    blocksize_row: int,
    blocksize_col: int,
    out: Optional[torch.Tensor] = None,
):
    # Function implementation here

def bsr_softmax(
    values: torch.Tensor,
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    blocksize_row: int,
    blocksize_col: int,
    out: Optional[torch.Tensor] = None,
):
    # Function implementation here
