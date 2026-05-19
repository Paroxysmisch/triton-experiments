import triton
import triton.language as tl
import torch

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q_ptr, k_ptr, g_ptr, A_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_km, stride_kk,
    stride_gz, stride_gh, stride_gm, stride_gk,
    stride_Az, stride_Ah, stride_Am, stride_Ak,
    BATCH, HEAD, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    bh = tl.program_id(2)
    mask_m = off_m < M
    mask_n = off_n < N

    # Compute pointers
    q_offs = bh * stride_qh
    k_offs = bh * stride_kh
    g_offs = bh * stride_gh
    A_offs = bh * stride_Ah

    q_block = tl.load(q_ptr + q_offs + off_m[:, None] * stride_qm + tl.arange(0, K)[None, :] * stride_qk, mask=(mask_m[:, None]), other=0.0)
    k_block = tl.load(k_ptr + k_offs + off_n[:, None] * stride_km + tl.arange(0, K)[None, :] * stride_kk, mask=(mask_n[:, None]), other=0.0)
    g_m = tl.load(g_ptr + g_offs + off_m * stride_gm, mask=mask_m, other=0.)[:, None]

    # Dot product q*k^T
    # shape: (BLOCK_M, K) x (K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
    b_A = tl.dot(q_block, tl.trans(k_block)) * SCALE
    # Gating
    b_A = b_A * g_m

    # Skip if i_i <= i_j to maintain upper-triangular or other causal structure
    for i in range(BLOCK_M):
        for j in range(BLOCK_N):
            if (off_m[i] <= off_n[j]):
                b_A[i, j] = tl.exp(b_A[i, j])
            else:
                b_A[i, j] = 0.0

    # Store
    tl.store(A_ptr + A_offs + off_m[:, None] * stride_Am + off_n[None, :] * stride_Ak, b_A, mask=(mask_m[:, None] & mask_n[None, :]))


@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q_ptr, k_ptr, g_ptr, A_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_km, stride_kk,
    stride_gz, stride_gh, stride_gm, stride_gk,
    stride_Az, stride_Ah, stride_Am, stride_Ak,
    BATCH, HEAD, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    bh = tl.program_id(2)
    mask_m = off_m < M
    mask_n = off_n < N

    q_offs = bh * stride_qh
    k_offs = bh * stride_kh
    g_offs = bh * stride_gh
    A_offs = bh * stride_Ah

    q_block = tl.load(q_ptr + q_offs + off_m[:, None] * stride_qm + tl.arange(0, K)[None, :] * stride_qk, mask=(mask_m[:, None]), other=0.0)
    k_block = tl.load(k_ptr + k_offs + off_n[:, None] * stride_km + tl.arange(0, K)[None, :] * stride_kk, mask=(mask_n[:, None]), other=0.0)
    g_m = tl.load(g_ptr + g_offs + off_m * stride_gm, mask=mask_m, other=0.)[:, None]

    b_A = tl.dot(q_block, tl.trans(k_block)) * SCALE
    b_A = b_A * g_m

    for i in range(BLOCK_M):
        for j in range(BLOCK_N):
            if (off_m[i] == off_n[j]):
                b_A[i, j] = tl.exp(b_A[i, j])
            else:
                b_A[i, j] = 0.0

    tl.store(A_ptr + A_offs + off_m[:, None] * stride_Am + off_n[None, :] * stride_Ak, b_A, mask=(mask_m[:, None] & mask_n[None, :]))


@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_split(
    q_ptr, k_ptr, g_ptr, A_intra_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_km, stride_kk,
    stride_gz, stride_gh, stride_gm, stride_gk,
    stride_Az, stride_Ah, stride_Am, stride_Ak,
    BATCH, HEAD, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
    SCALE: tl.constexpr
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    split_id = tl.program_id(2)
    bh = tl.program_id(3)

    mask_m = off_m < M
    mask_n = off_n < N

    q_offs = bh * stride_qh
    k_offs = bh * stride_kh
    g_offs = bh * stride_gh
    A_offs = bh * stride_Ah

    # Range for partial K
    k_start = split_id * (K // SPLIT_K)
    k_end = tl.minimum(K, (split_id + 1) * (K // SPLIT_K))
    k_range = k_end - k_start

    # Load partial blocks
    q_block = tl.load(
        q_ptr + q_offs + off_m[:, None] * stride_qm + tl.arange(k_start, k_end)[None, :] * stride_qk,
        mask=(mask_m[:, None] & (tl.arange(k_start, k_end)[None, :] < K)), other=0.0
    )
    k_block = tl.load(
        k_ptr + k_offs + off_n[:, None] * stride_km + tl.arange(k_start, k_end)[None, :] * stride_kk,
        mask=(mask_n[:, None] & (tl.arange(k_start, k_end)[None, :] < K)), other=0.0
    )
    g_m = tl.load(g_ptr + g_offs + off_m * stride_gm, mask=mask_m, other=0.)[:, None]

    b_A_partial = tl.dot(q_block, tl.trans(k_block)) * SCALE
    b_A_partial = b_A_partial * g_m

    # Write partial results to A_intra
    tl.store(
        A_intra_ptr + A_offs + (split_id * M + off_m)[:, None] * stride_Am + off_n[None, :] * stride_Ak,
        b_A_partial,
        mask=(mask_m[:, None] & mask_n[None, :])
    )


@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge(
    A_intra_ptr, A_ptr,
    stride_Aintra_z, stride_Aintra_m, stride_Aintra_k,
    stride_Az, stride_Am, stride_Ak,
    BATCH, HEAD, M, N, SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    bh = tl.program_id(2)
    mask_m = off_m < M
    mask_n = off_n < N

    A_intra_offs = bh * stride_Aintra_z
    A_offs = bh * stride_Az

    b_out = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for split_id in range(SPLIT_K):
        partial = tl.load(
            A_intra_ptr + A_intra_offs + (split_id * M + off_m)[:, None] * stride_Aintra_m + off_n[None, :] * stride_Aintra_k,
            mask=(mask_m[:, None] & mask_n[None, :]),
            other=0.0
        )
        b_out += partial

    # Exponential step or any final transform if needed
    for i in range(BLOCK_M):
        for j in range(BLOCK_N):
            b_out[i, j] = tl.exp(b_out[i, j])

    tl.store(A_ptr + A_offs + off_m[:, None] * stride_Am + off_n[None, :] * stride_Ak, b_out, mask=(mask_m[:, None] & mask_n[None, :]))


@triton.jit
def chunk_gla_fwd_kernel_o(
    A_ptr, g_ptr, o_ptr, v_ptr,
    stride_Az, stride_Am, stride_Ak,
    stride_gz, stride_gm,
    stride_oz, stride_om, stride_ok,
    stride_vz, stride_vm, stride_vk,
    BATCH, HEAD, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    bh = tl.program_id(1)
    mask_m = off_m < M

    A_offs = bh * stride_Az
    g_offs = bh * stride_gz
    o_offs = bh * stride_oz
    v_offs = bh * stride_vz

    # Accumulate over N dimension
    accum = tl.zeros([BLOCK_M, K], dtype=tl.float32)
    for col_block in range(0, N, BLOCK_N):
        off_n = col_block + tl.arange(0, BLOCK_N)
        mask_n = off_n < N
        A_block = tl.load(
            A_ptr + A_offs + off_m[:, None] * stride_Am + off_n[None, :] * stride_Ak,
            mask=(mask_m[:, None] & mask_n[None, :]),
            other=0.0
        )
        v_block = tl.load(
            v_ptr + v_offs + off_n[:, None] * stride_vm + tl.arange(0, K)[None, :] * stride_vk,
            mask=(mask_n[:, None]),
            other=0.0
        )
        accum += tl.dot(A_block, v_block)

    # Gating
    g_m = tl.load(g_ptr + g_offs + off_m * stride_gm, mask=mask_m, other=0.)[:, None]
    accum = accum * g_m

    # Store result
    tl.store(o_ptr + o_offs + off_m[:, None] * stride_om + tl.arange(0, K)[None, :] * stride_ok, accum, mask=(mask_m[:, None]))


def chunk_fwd_intra_gated_gk_fn(
    q, k, g, A,
    BATCH, HEAD, M, N, K, BLOCK_M, BLOCK_N,
    scale, split_k,
    stream=None
):
    grid_intra_sub_inter = ( (M + BLOCK_M - 1)//BLOCK_M, (N + BLOCK_N - 1)//BLOCK_N, HEAD*BATCH )
    chunk_gla_fwd_A_kernel_intra_sub_inter[grid_intra_sub_inter](
        q, k, g, A,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k
