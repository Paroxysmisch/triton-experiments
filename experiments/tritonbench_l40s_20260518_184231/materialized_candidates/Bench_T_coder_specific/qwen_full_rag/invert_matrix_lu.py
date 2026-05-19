import torch
import triton
import triton.language as tl


@triton.jit
def invert_triangular(
    lower,
    upper,
    batch_dims,
    n,
    dtype,
    uplo: tl.constexpr,
    unit_diagonal: tl.constexpr,
    trans: tl.constexpr,
    batch_stride,
    stride_l,
    stride_u,
    OUT,
    o_batch_stride,
    o_stride_l,
    o_stride_u,
    TM: tl.constexpr,
    TN: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    b_offset = pid_b * batch_stride
    i_m = pid_m * TM + tl.arange(0, TM)
    mask_m = i_m < n
    i_n = tl.arange(0, TN)
    mask_n = i_n < n

    if trans:
        l_ptrs = (
            lower
            + b_offset
            + i_m[:, None] * stride_l
            + i_n[None, :] * 1
            + (1 - uplo) * (n - 1) * stride_l
        )
        u_ptrs = (
            upper
            + b_offset
            + i_m[:, None] * 1
            + i_n[None, :] * stride_u
            + (uplo - 1) * (n - 1) * stride_u
        )
    else:
        l_ptrs = (
            lower
            + b_offset
            + i_m[:, None] * stride_l
            + i_n[None, :] * stride_l
            + (1 - uplo) * (n - 1) * stride_l
        )
        u_ptrs = (
            upper
            + b_offset
            + i_m[:, None] * stride_u
            + i_n[None, :] * stride_u
            + (uplo - 1) * (n - 1) * stride_u
        )

    l = tl.load(l_ptrs, mask=(mask_m[:, None]) & (mask_n[None, :]), other=dtype.one)
    u = tl.load(u_ptrs, mask=(mask_m[:, None]) & (mask_n[None, :]), other=dtype.zero)

    if unit_diagonal:
        l = l - tl.diag(l)

    l_inv = tl.triangular_solve(l, dtype.one, lower=(not uplo), unit_diagonal=True)
    u_inv = tl.triangular_inverse(u, lower=(uplo))

    if trans:
        y = (
            u_inv
            @ l_inv.T
            if uplo
            else l_inv.T @ u_inv
            # upper, T @ lower == lower @ T
        )
    else:
        y = (
            u_inv @ l_inv
            if uplo
            else l_inv @ u_inv
            # upper, l @ lower == lower @ l
        )

    o_y_strides = (
        o_batch_stride
        + tl.arange(0, TM)[:, None] * o_stride_l
        + tl.arange(0, TN)[None, :] * o_stride_u
    )
    o_ptr = OUT + b_offset + o_y_strides
    tl.store(o_ptr, y, mask=mask_m[:, None] & mask_n[None, :])


def invert_matrix_lu(
    A: torch.Tensor,
    *,
    pivot: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the inverse of a square matrix using LU decomposition.
    A = P L U
    A^-1 = U^-1 L^-1 P

    Args:
        A: Square invertible matrix.
            Must be of shape [..., n, n].
        pivot: Flag for indicating whether partial pivoting should be used.
            Default is True.
        out: Optional output buffer.
            If provided, output is written to this tensor instead of a newly allocated one.
            The buffer must be of shape [..., n, n].

    Returns:
        Inverse of A.
    """
    device = A.device
    dtype = A.dtype
    assert A.ndim >= 2
    n = A.shape[-1]
    assert A.shape[-2] == n

    if A.stride(-1) != 1:
        A = A.contiguous()

    if pivot:
        # TODO: add support for Strassen's algorithm.
        raise NotImplementedError("Partial pivoting requires Strassen's algorithm.")
    else:
        # https://comeonecodeaday.wordpress.com/2018/07/25/matrix-inversion-in-lu-form/
        # Instead of partial pivoting, we can improve numerical stability
        # by rearranging the rows of A into blocks of uniformly distributed row norms.
        # This is equivalent to multiplying A by a permutation matrix P,
        # such that P A is block-diagonal with blocks having uniformly distributed row norms.
        # The product P^-1 can then be computed efficiently, and combined with L and U
        # to form the final inverse.
        pass

    batch_shape = A.shape[:-2]
    batch_size = torch.tensor(batch_shape).numel()
    batch_stride = A.strides[-3] if len(batch_shape) >= 3 else 0

    if len(batch_shape) >= 3:
        A = A.view(batch_size, -1, n, n)
    else:
        A = A.unsqueeze(0)

    info = torch.empty((batch_size,), dtype=torch.int32, device=device)
    buf_lu = A.clone()
    torch.lu_factor(A, pivot, info, get_infos=False, out=buf_lu)

    buf_l, buf_u, _ = torch.lu_unpack(buf_lu, pivot)

    if out is None:
        out = torch.empty_like(A)
    else:
        assert out.shape == A.shape
        assert out.stride(-1) == 1

    if len(batch_shape) >= 3:
        out = out.view(batch_size, -1, n, n)
    else:
        out = out.unsqueeze(0)

    o_batch_stride = out.strides[-4] if len(batch_shape) >= 4 else 0
    o_l_stride = out.strides[-3]
    o_u_stride = out.strides[-2]

    def grid(META):
        return (
            triton.cdiv(n, META["TM"]),
            batch_size,
        )

    invert_triangular[grid](
        buf_l,
        buf_u,
        batch_dims=batch_shape,
        n=n,
        dtype=dtype,
        uplo=0,
        unit_diagonal=False,
        trans=False,
        batch_stride=batch_stride,
        stride_l=o_l_stride,
        stride_u=o_u_stride,
        OUT=out,
        o_batch_stride=o_batch_stride,
        o_stride_l=out.strides[-3],
        o_stride_u=out.strides[-2],
    )

    if len(batch_shape) >= 3:
        out = out.view(*batch_shape, n, n)
    else:
        out = out.squeeze(0)

    return out
