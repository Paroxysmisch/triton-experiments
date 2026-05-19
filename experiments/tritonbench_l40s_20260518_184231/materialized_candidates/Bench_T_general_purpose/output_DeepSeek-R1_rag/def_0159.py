import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(
    A_ptr,
    L_ptr,
    n,
    stride_ba,
    stride_ma,
    stride_na,
    stride_bl,
    stride_ml,
    stride_nl,
    err_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid
    offs_m = tl.arange(0, BLOCK_SIZE)
    mask_m = offs_m < n

    A_ptr += batch_idx * stride_ba + offs_m[:, None] * stride_ma + offs_m[None, :] * stride_na
    L_ptr += batch_idx * stride_bl
    err_ptr += batch_idx

    for i in range(0, n):
        sum_sq = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        for k in range(0, i):
            l_ik = tl.load(L_ptr + k * stride_ml + i * stride_nl, mask=mask_m, other=0.0)
            sum_sq += l_ik * l_ik

        a_ii = tl.load(A_ptr + i * stride_ma + i * stride_na, mask=mask_m, other=0.0)
        l_ii = tl.sqrt(a_ii - sum_sq)
        tl.store(L_ptr + i * stride_ml + i * stride_nl, l_ii, mask=mask_m)

        if tl.any(l_ii <= 0.0):
            tl.atomic_xchg(err_ptr, 1)

        for j in range(i + 1, n):
            sum_l = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
            for k in range(0, i):
                l_jk = tl.load(L_ptr + k * stride_ml + j * stride_nl, mask=mask_m, other=0.0)
                l_ik = tl.load(L_ptr + k * stride_ml + i * stride_nl, mask=mask_m, other=0.0)
                sum_l += l_jk * l_ik

            a_ji = tl.load(A_ptr + j * stride_ma + i * stride_na, mask=mask_m, other=0.0)
            l_ji = (a_ji - sum_l) / l_ii
            tl.store(L_ptr + i * stride_ml + j * stride_nl, l_ji, mask=mask_m)

@triton.jit
def conjugate_transpose_kernel(
    L_ptr,
    U_ptr,
    n,
    stride_bl,
    stride_ml,
    stride_nl,
    stride_bu,
    stride_mu,
    stride_nu,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid
    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, BLOCK_SIZE)
    mask = (offs_m < n)[:, None] & (offs_n < n)[None, :]

    L_ptr += batch_idx * stride_bl + offs_m[:, None] * stride_ml + offs_n[None, :] * stride_nl
    U_ptr += batch_idx * stride_bu + offs_n[:, None] * stride_mu + offs_m[None, :] * stride_nu

    l = tl.load(L_ptr, mask=mask, other=0.0)
    u = tl.trans(l)
    if L_ptr.dtype.element_ty != tl.float32 and L_ptr.dtype.element_ty != tl.float64:
        u = tl.conj(u)
    tl.store(U_ptr, u, mask=mask)

def linalg_cholesky(A, *, upper=False, out=None):
    assert A.shape[-1] == A.shape[-2], "Input must be square matrices"
    A = A.contiguous()
    dtype = A.dtype
    device = A.device
    n = A.shape[-1]
    batch_dims = A.shape[:-2]
    num_batches = 1 if not batch_dims else int(torch.prod(torch.tensor(batch_dims)))
    A_flat = A.view(num_batches, n, n)
    L = torch.empty_like(A_flat) if out is None else out.view(num_batches, n, n)
    err = torch.zeros(1, dtype=torch.int32, device=device)

    BLOCK_SIZE = triton.next_power_of_2(n)
    BLOCK_SIZE = max(min(BLOCK_SIZE, 128), 16)

    grid = (num_batches,)
    cholesky_kernel[grid](
        A_flat, L, n,
        A_flat.stride(0), A_flat.stride(1), A_flat.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        err,
        BLOCK_SIZE=BLOCK_SIZE
    )

    if device.type == 'cuda':
        torch.cuda.synchronize()

    if err.item() != 0:
        raise RuntimeError("Cholesky decomposition failed: matrix is not positive-definite.")

    if upper:
        U = torch.empty_like(L) if out is None else out.view(num_batches, n, n)
        conjugate_transpose_kernel[grid](
            L, U, n,
            L.stride(0), L.stride(1), L.stride(2),
            U.stride(0), U.stride(1), U.stride(2),
            BLOCK_SIZE=BLOCK_SIZE
        )
        L = U

    if out is not None:
        out.copy_(L.view(*A.shape))
    return L.view(*A.shape)
