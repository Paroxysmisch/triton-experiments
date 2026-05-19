,
    alpha=1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if input.dtype is torch.bool:
        raise RuntimeError("sampled_addmm: expected non-boolean input")
    if not is_bsr_shape(input.shape):
        raise ValueError(
            f"sampled_addmm: expected a BSR shaped input tensor but got: {input.shape}"
        )
    if input.ndim != mat1.ndim or input.ndim != mat2.ndim:
        raise ValueError(
            f"sampled_addmm: expected input, mat1, mat2 to have same number of dimensions "
            f"(3 are given), but got shapes {input.shape}, {mat1.shape}, {mat2.shape}"
        )
    if input.ndim > 3:
        if input.shape[-2] != mat1.shape[-2] or input.shape[-2] != mat2.shape[-3]:
            raise ValueError(
                f"sampled_addmm: incompatible dimensions for input and mat1/mat2 "
                f"(input.shape[-2] == mat1.shape[-2] == mat2.shape[-3] "
                f"expected, but got {input.shape}, {mat1.shape}, {mat2.shape}"
            )

    if out is None:
        out = torch.empty_like(input)
    elif out.dtype is torch.bool:
        raise RuntimeError("sampled_addmm: expected non-boolean out")
    elif out.shape != input.shape:
        raise ValueError(
            f"sampled_addmm: expected out to have shape same as input "
            f"(GOT: {out.shape}, EXPECTED: {input.shape})"
        )
    elif not is_bsr_shape(out.shape):
        raise ValueError(
            f"sampled_addmm: expected a BSR shaped out tensor but got: {out.shape}"
        )

    k = mat1.shape[-1]
    if k != mat2.shape[-2]:
        raise ValueError(
            f"sampled_addmm: incompatible dimensions for mat1 and mat2 "
            f"(mat1.shape[-1] == mat2.shape[-2] expected, but got {mat1.shape}, {mat2.shape}"
        )

    if k < 1:
        raise ValueError(
            "sampled_addmm: last dimension of mat1 must be equal to the second "
            "dimension of mat2 (inferred to be {} from the shapes of both tensors)".format(k)
        )

    if not isinstance(alpha, Number) or not isinstance(beta, Number):
        raise TypeError(
            "sampled_addmm: `alpha` and `beta` must be numbers"
        )

    if alpha == 0:
        return out

    IS_BETA_ZERO = beta == 0

    if input.dtype is torch.bool:
        acc_dtype = tl.int32
    else:
        acc_dtype = tl.float32 if alpha == 1 else tl.float16

    BLOCKSIZE_ROW, BLOCKSIZE_COL = get_block_shape(mat1)

    # number of TILE_K such that one block of mat1 is always in one TILE.
    TILE_K = int(
        triton.next_power_of_2(k) // max(1, mat1.shape[-1] // (BLOCKSIZE_ROW * BLOCKSIZE_COL))
    )

    grid = lambda META: (
        triton.cdiv(META["num_row_blocks"], META["BLOCKSIZE_ROW"]),
        input.size(0),
    )

    _sampled_addmm_kernel[grid](
        alpha,
        beta,
        IS_BETA_ZERO,
        BLOCKSIZE_ROW,
        BLOCKSIZE_COL,
        k,
        TILE_K,
        input,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        input.crow_indices(),
        input.crow_indices().stride(0),
        input.crow_indices().stride(1),
        input.col_indices(),
        input.col_indices().stride(0),
        input.col_indices().stride(1),
        mat1,
        mat1.stride(0),
        mat1.stride(1),
        mat1.stride(2),
        mat1.stride(3),
        mat2,
        mat2.stride(0),
        mat2.stride(1),
        mat2.stride(2),
        mat2.stride(3),
        acc_dtype=acc_dtype,
        allow_tf32=mat1.dtype.is_tf32_compatible(),
    )

    return out
