, values, crow_indices, col_indices, dense, output,
    batch_shape, batch_stride, crow_indices_stride, col_indices_stride,
    dense_stride, output_stride, num_warps, num_stages, nproc
):
    # Function implementation here

def _run_softmax_kernel(
    blocksize, values, crow_indices, num_batches, num_rows, max_row_nnz,
    num_warps, num_stages
):
    # Function implementation here

def sampled_addmm(
    input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor,
    *, beta=1, alpha=1, out: Optional[torch.Tensor] = None,
    skip_checks: bool = False, max_grid: Optional[Tuple[int, int, int]] = None
):
    # Function implementation here

def bsr_strided_dense_matmul(
    values: torch.Tensor, crow_indices: torch.Tensor, col_indices: torch.Tensor,
    dense: torch.Tensor, sparsity_block: int, block_strides: Tuple[int, int, int],
    out: Optional[torch.Tensor] = None, max_grid: Optional[Tuple[int, int, int]] = None
):
    # Function implementation here

def bsr_softmax(input: torch.Tensor, sparsity_block: int,
                out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Function implementation here
