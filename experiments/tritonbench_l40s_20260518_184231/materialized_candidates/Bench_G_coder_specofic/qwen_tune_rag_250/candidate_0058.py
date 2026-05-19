_stride,
    output_row_block_stride,
    output_col_block_stride,
    block_dummy_id,
    acc_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    # Kernel implementation here

@triton.jit
def _bsr_softmax_kernel(
    crow_indices_ptr,
    crow_indices_batch_stride,
    crow_indices_stride,
    col_indices_ptr,
    col_indices_batch_stride,
    col_indices_stride,
    values_ptr,
    values_batch_stride,
    values_nnz_stride,
    values_row_block_stride,
    values_col_block_stride,
    input_row_block,
    input_col_block,
    MAX_TILE_DIM: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
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
    block_dummy_id: int = 0,
    out: Optional[torch.Tensor] = None,
):
    # Function implementation here

def bsr_softmax(values, crow_indices, col_indices, dim=None, dtype=None):
    # Function implementation here
