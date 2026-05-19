_stride,
    values_nnz_stride,
    values_row_block_stride,
    values_col_block_stride,
    norm_row_ptr,
    norm_row_batch_stride,
    norm_row_stride,
    BLOCKSIZE_ROW: tl.constexpr,
    BLOCKSIZE_COL: tl.constexpr,
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
    norm_row: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
):
    # Function implementation here

def bsr_softmax(
    values: torch.Tensor,
    crow_indices: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
):
    # Function implementation here
