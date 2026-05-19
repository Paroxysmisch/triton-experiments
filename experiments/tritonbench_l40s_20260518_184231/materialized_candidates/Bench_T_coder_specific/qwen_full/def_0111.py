import torch
import triton
import triton.language as tl

@triton.jit
def _low_rank_svd_approximation(
    A,
    U,
    S,
    V,
    stride_a_batch,
    stride_a_m,
    stride_a_n,
    stride_u_batch,
    stride_u_m,
    stride_u_k,
    stride_s_batch,
    stride_s_k,
    stride_v_batch,
    stride_v_k,
    stride_v_n,
    k,
    full_matrices: tl.constexpr,
    batch_shape,
    m,
    n,
    bmm_m,
    bmm_k,
    bmm_n,
    dot_k,
    dot_n,
    dot_m,
    output_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = tl.arange(0, BLOCK_SIZE_N)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    batch_offset_a = pid_batch * stride_a_batch
    batch_offset_u = pid_batch * stride_u_batch
    batch_offset_s = pid_batch * stride_s_batch
    batch_offset_v = pid_batch * stride_v_batch
    offset_mask_m = offset_m < m
    offset_mask_n = offset_n < n
    offset_mask_k = offset_k < k

    a_ptrs = (
        A
        + batch_offset_a
        + (offset_m[:, None] * stride_a_m + offset_k[None, :] * stride_a_n)
    )
    u_ptrs = (
        U
        + batch_offset_u
        + (offset_m[:, None] * stride_u_m + offset_k[None, :] * stride_u_k)
    )
    s_ptrs = S + batch_offset_s + offset_k
    v_ptrs = (
        V
        + batch_offset_v
        + (offset_k[:, None] * stride_v_k + offset_n[None, :] * stride_v_n)
    )

    if full_matrices:
        A = tl.load(a_ptrs, mask=offset_mask_m[:, None] & offset_mask_k[None, :])
        U = tl.load(u_ptrs, mask=offset_mask_m[:, None] & offset_mask_k[None, :])
        V = tl.load(v_ptrs, mask=offset_mask_k[:, None] & offset_mask_n[None, :])
    else:
        A = tl.load(a_ptrs, mask=offset_mask_m[:, None] & offset_mask_k[None, :])
        U = tl.load(u_ptrs, mask=offset_mask_m[:, None] & offset_mask_k[None, :])
        V = tl.load(v_ptrs, mask=offset_mask_k[:, None] & offset_mask_n[None, :])

    Sigma = tl.load(s_ptrs, mask=offset_mask_k).to(tl.float32)
    zero = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    A = A.to(output_dtype)
    Sigma = Sigma.to(output_dtype)
    U = U.to(output_dtype)
    V = V.to(output_dtype)

    U, Sigma, V = tl.svd(A, full_matrices=full_matrices)

    U = U.to(output_dtype)
    Sigma = Sigma.to(output_dtype)
    V = V.to(output_dtype)

    for i in range(0, tl.cdiv(n, BLOCK_SIZE_N)):
        v_chunk_ptrs = (
            V
            + batch_offset_v
            + (offset_k[:, None] * stride_v_k + (offset_n + i * BLOCK_SIZE_N)[None, :] * stride_v_n)
        )
        v_chunk = tl.load(v_chunk_ptrs, mask=offset_mask_k[:, None] & ((offset_n + i * BLOCK_SIZE_N)[None, :] < n))
        zero[:, i * BLOCK_SIZE_N:(i + 1) * BLOCK_SIZE_N] = v_chunk
    V = zero

    S = Sigma

    tl.store(u_ptrs, U, mask=offset_mask_m[:, None] & offset_mask_k[None, :])
    tl.store(s_ptrs, S, mask=offset_mask_k)
    tl.store(v_ptrs, V, mask=offset_mask_k[:, None] & offset_mask_n[None, :])


def low_rank_svd_approximation(A, k, *, full_matrices=True, out=None) -> torch.Tensor:
    batch_shape, m, n = A.shape
    batch_dim = batch_shape.numel()
    A = A.reshape(batch_dim, m, n)
    if out is None:
        U = torch.empty_like(A, dtype=A.dtype)
        S = torch.empty((batch_dim, k), dtype=torch.float32, device=A.device)
        V = torch.empty_like(A, dtype=A.dtype)
    else:
        U, S, V = out
    bmm_m, bmm_k = (1, k) if full_matrices else (m, k)
    bmm_k, bmm_n = k, n
    dot_k, dot_n = k, 1
    dot_m, dot_k = 1, k
    _low_rank_svd_approximation[(batch_dim,)](
        A,
        U,
        S,
        V,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        U.stride(0),
        U.stride(1),
        U.stride(2),
        S.stride(0),
        S.stride(1),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        k,
        full_matrices,
        batch_shape,
        m,
        n,
        bmm_m,
        bmm_k,
        bmm_n,
        dot_k,
        dot_n,
        dot_m,
        A.dtype,
        BLOCK_SIZE_M=triton.next_power_of_2(m),
        BLOCK_SIZE_N=triton.next_power_of_2(n),
        BLOCK_SIZE_K=triton.next_power_of_2(k),
        num_warps=8,
    )
    return U, S, V
