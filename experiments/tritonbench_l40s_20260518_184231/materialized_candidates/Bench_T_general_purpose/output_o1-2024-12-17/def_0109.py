import torch
import triton
import triton.language as tl

@triton.jit
def _scale_rows_kernel(
    U_ptr, S_ptr, OUT_ptr,
    stride_u1, stride_u2, stride_out1, stride_out2,
    n_rows, n_cols,
    BLOCK_M: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = tl.arange(0, BLOCK_M)
    row_ids = row_start + rows
    mask_rows = row_ids < n_rows

    # Loop over columns
    for col in range(n_cols):
        row = row_ids
        s_val = tl.load(S_ptr + row, mask=mask_rows, other=0.0)
        u_val = tl.load(U_ptr + row * stride_u1 + col * stride_u2, mask=mask_rows, other=0.0)
        out_val = s_val * u_val
        tl.store(OUT_ptr + row * stride_out1 + col * stride_out2, out_val, mask=mask_rows)


@triton.jit
def _gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_a1, stride_a2,
    stride_b1, stride_b2,
    stride_c1, stride_c2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm_mask = rm < M
    rn_mask = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Loop over K dimension
    for kk in range(0, K, BLOCK_K):
        rk = tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (rm[:, None] * stride_a1 + (kk + rk[None, :]) * stride_a2)
        b_ptrs = B_ptr + ((kk + rk[:, None]) * stride_b1 + rn[None, :] * stride_b2)
        a = tl.load(a_ptrs, mask=(rm_mask[:, None] & (kk + rk[None, :] < K)), other=0.0)
        b = tl.load(b_ptrs, mask=((kk + rk[:, None] < K) & rn_mask[None, :]), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (rm[:, None] * stride_c1 + rn[None, :] * stride_c2)
    tl.store(c_ptrs, acc, mask=(rm_mask[:, None] & rn_mask[None, :]))


def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None):
    # Perform SVD via torch
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    # Compute threshold
    threshold = rcond * S.max(dim=-1, keepdim=True).values
    # Mask out small singular values
    S_inv = torch.where(S > threshold, 1.0 / S, torch.zeros_like(S))

    # Transpose U to get U^H (conjugate transpose for complex)
    Uh = U.transpose(-2, -1).conj()

    # Scale rows of U^H by S_inv (diag(S_inv)*U^H)
    # (U^H shape: [n, m], S_inv shape: [n])
    n, m = Uh.shape[-2], Uh.shape[-1]
    scaled = torch.empty_like(Uh)
    grid = lambda META: ( (n + META['BLOCK_M'] - 1) // META['BLOCK_M'], )
    _scale_rows_kernel[grid](
        Uh, S_inv,
        scaled,
        Uh.stride(-2), Uh.stride(-1),
        scaled.stride(-2), scaled.stride(-1),
        n, m,
        BLOCK_M=128
    )

    # Now compute V^H @ scaled (this is V^H * (diag(S_inv)*U^H))
    # V^H shape: [n, n], scaled shape: [n, m] -> out shape: [n, m]
    nVH, nVH2 = Vh.shape
    M, N = nVH, nVH2  # for naming consistency
    assert M == n, "Dimension mismatch in V^H"
    out_shape = (n, m)
    if out is None:
        out = torch.empty(out_shape, dtype=A.dtype, device=A.device)

    grid_m = ( (n + 128 - 1) // 128 )
    grid_n = ( (m + 128 - 1) // 128 )
    _gemm_kernel[(grid_m, grid_n)](
