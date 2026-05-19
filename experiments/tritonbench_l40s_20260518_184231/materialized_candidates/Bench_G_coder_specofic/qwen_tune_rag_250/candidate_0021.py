dense_col_block_stride,
    output_ptr,
    output_batch_stride,
    output_tiled_row_stride,
    output_tiled_col_stride,
    output_row_block_stride,
    output_col_block_stride,
    nnz,
    mat1_tiled_row,
    mat1_tiled_col,
    mat2_tiled_row,
    mat2_tiled_col,
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
    k,
    BLOCKSIZE_ROW: tl.constexpr,
    BLOCKSIZE_COL: tl.constexpr,
    acc_dtype: tl.constexpr,
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
    f_name = "sampled_addmm"

    check_bsr_layout(f_name, input)
    input_broadcasted = broadcast_batch_dims_bsr(f_name, input, mat1, mat2)

    if not skip_checks:
        check_device(f_name, mat1, input.device)
        check_device(f_name, mat2, input.device)
        if beta != 0.0 and input.dtype is torch.bool:
            check(
                False,
                f"{f_name}(): having beta == {beta} not equal to 0.0 with boolean mask is not allowed."
            )
        if input.dtype is not torch.bool:
            check_dtype(f_name, mat1, input.dtype)
            check_dtype(f_name, mat2, input.dtype)
        else:
            check_dtype(f_name, mat1, mat2.dtype)
        check_mm_compatible_shapes(f_name, mat1, mat2)
        if out is not None:
            check_bsr_layout(f_name, out)
            check_device(f_name, out, mat1.device)
            check_dtype(f_name, out, input.dtype)
            check(
                out.shape == input_broadcasted.shape
                and out._nnz() == input._nnz(),
                f"{f_name}(): Expects `out` to be of shape {input_broadcasted.shape} "
                f"and with nnz equal to {input_broadcasted._nnz()} "
                f"but got out.shape = {out.shape} and out.nnz = {out._nnz()}"
            )

    if out is None:
        out = input_broadcasted.to(mat1.dtype, copy=True)
    else:
        out.copy_(input_broadcasted)

    if out.numel() == 0 or out._nnz() == 0:
        return out

    blocksize = out.values().shape[-2:]
    m = mat1.size(-2)
    n = mat2.size(-1)
    k = mat1.size(-1)

    mat1 = mat1.detach()
    mat2 = mat2.detach()

    mat1 = tile_to_blocksize(mat1, (m, k), blocksize)
    mat2 = tile_to_blocksize(mat2, (k, n), blocksize)

    nnz = out._nnz()
    crow_indices = out.crow_indices()
    col_indices = out.col_indices()

    def grid(META):
        return (triton.cdiv(crow_indices[-1].item() - crow_indices[0].item(), META['BLOCKSIZE_ROW']),
                triton.cdiv(k, META['BLOCK_K']))

    def num_warps():
        if blocksize[0] <= 32 and blocksize[1] <= 32:
            return 4
        else:
            return 8

    _sampled_addmm_kernel[grid](
        alpha, beta, beta == 0.0,
        blocksize[0], blocksize[1],
        k, TILE_K=16,  # type: ignore
        values_ptr=out.values(),
        values_batch_stride=out.values().stride(0),
        values_nnz_stride=out.values().stride(1),
        values_row_block_stride=out.values().stride(2),
        values_col_block_stride=out.values().stride(3),
        crow_indices_ptr=crow_indices,
        crow_indices_batch_stride=crow_indices.stride(0),
        crow_indices_stride=crow_indices.stride(1),
        col_indices_ptr=col_indices,
        col_indices_batch_stride=col_indices.stride(0),
        col_indices_stride=col_indices.stride(1),
        mat1_ptr=mat1,
        mat1_batch_stride=mat1.stride(0),
        mat1_tiled_row_stride=mat1.stride(1),
        mat1_tiled_col_stride=mat1.stride(2),
        mat1_row_block_stride=mat1.stride(3),
        mat1_col_block_stride=mat1.stride(4),
        mat2_ptr=mat2,
        mat2_batch_stride=mat2.stride(0),
        mat2_tiled_row_stride=mat2.stride(1),
        mat2_tiled_col_stride=mat2.stride(2),
        mat2_row_block_stride=mat2.stride(3),
        mat2_col_block_stride=mat2.stride(4),
        acc_dtype=tl.bfloat16 if mat1.dtype is torch.bfloat16 else tl.float32,
        allow_tf32=True,
        num_warps=num_warps(),
    )
    return out

def bsr_strided_dense_rowspace(
    values: torch.Tensor,
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    dense: torch.Tensor,
    blocksize: Tuple[int, int],
    out: Optional[torch.Tensor] = None,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
):
    f_name = "bsr_strided_dense_rowspace"
    check_device(f_name, crow_indices, values.device)
    check_device(f_name, col_indices, values.device)
    check_device(f_name, dense, values.device)
    check_dense_layout(f_name, dense)
    check_bsr_layout(f_name, values)

    if out is not None:
        check_device(f_name, out, values.device)
        check_dtype(f_name, out, values.dtype)
        check(
            out.shape == dense.shape
            and out._nnz() == values._nnz(),
            f"{f_name}(): Expects `out` to be of shape {dense.shape} "
            f"and with nnz equal to {values._nnz()} "
            f"but got out.shape = {out.shape} and out.nnz = {out._nnz()}"
        )

    if out is None:
        out = torch.empty_like(dense, dtype=values.dtype, memory_format=torch.contiguous_format)

    if out.numel() == 0 or out._nnz() == 0:
        return out

    nnz = values._nnz()
    m = crow_indices.size(-1) - 1
    k = values.size(-1)
    n = dense.size(-1)

    crow_indices = crow_indices.detach()
    col_indices = col_indices.detach()
    values = values.detach()
    dense = dense.detach()

    crow_indices = crow_indices.expand(nnz)
    col_indices = col_indices.expand(nnz)
    values = values.squeeze_()
    dense = tile_to_blocksize(dense, (m, n), blocksize)
    out = tile_to_blocksize(out, (m, n), blocksize)

    def grid(META):
        return (triton.cdiv(m, META['BLOCKSIZE_ROW']),
                triton.cdiv(n, META['BLOCKSIZE_COL']),
                1)

    def num_warps():
        if blocksize[0] <= 32 and blocksize[1] <= 32:
            return 4
        else:
            return 8

    _bsr_strided_dense_rowspace_kernel[grid](
        blocksize[0], blocksize[1],
        values_ptr=values,
        values_batch_stride=values.stride(0),
        values_nnz_stride=values.stride(1),
        values_row_block_stride=values.stride(2),
        values_col_block_stride=values.stride(3),
        crow_indices_ptr=crow_indices,
        crow_indices_batch_stride=crow_indices.stride(0),
        crow_indices_stride=crow_indices.stride(1),
        col_indices_ptr=col_indices,
        col_indices_batch_stride=col_indices.stride(0),
        col_indices_stride=col_indices.stride(1),
        dense_ptr=dense,
        dense_batch_stride=dense.stride(0),
        dense_tiled_row_stride=dense.stride(1),
        dense_tiled_col_stride=dense.stride(2),
        dense_row_block_stride=dense.stride(3),
        dense_col_block_stride=dense.stride(4),
        output_ptr=out,
        output_batch_stride=out.stride(0),
        output_tiled_row_stride=out.stride(1),
        output_tiled_col_stride
