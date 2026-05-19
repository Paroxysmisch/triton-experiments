import triton
import triton.language as tl

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q_ptr, k_ptr, g_ptr, A_ptr,
    stride_qz, stride_qh, stride_qm,
    stride_kz, stride_kh, stride_kn,
    stride_gz, stride_gh, stride_gm,
    stride_Az, stride_Am, stride_An,
    M, N, K, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # We consider a single "batch" dimension for demonstration
    # Ex: usage in multi-head attn can expand relevant dims

    # Initialize accumulators
    b_A = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    offs_k = tl.arange(0, BLOCK_K)
    for k_offs in range(0, K, BLOCK_K):
        k_mask = k_offs + offs_k < K
        # Load Q, K, G sub-block
        q_block = tl.load(
            q_ptr + offs_m[:, None]*stride_qm + (k_offs + offs_k[None, :])*stride_qh,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0
        )
        k_block = tl.load(
            k_ptr + offs_n[None, :]*stride_km + (k_offs + offs_k[:, None])*stride_kh,
            mask=(offs_n[None, :] < N) & k_mask[:, None],
            other=0.0
        )
        g_block = tl.load(
            g_ptr + offs_m[:, None]*stride_gm + (k_offs + offs_k[None, :])*stride_gh,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0
        )

        # Weighted dot product
        prod = tl.dot(q_block, k_block)
        # Gating
        gate = tl.dot(g_block, k_block)

        # Scale & exponentiate
        attn = tl.exp((prod + gate) * scale)

        # Accumulate
        b_A += attn

    # Write out to A (upper-triangular skip if i <= j for demonstration)
    mask_store = (offs_m[:, None] < M) & (offs_n[None, :] < N) & (offs_m[:, None] > offs_n[None, :])
    tl.store(
        A_ptr + offs_m[:, None]*stride_Am + offs_n[None, :]*stride_An,
        b_A.to(tl.float32),
        mask=mask_store
    )

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q_ptr, k_ptr, g_ptr, A_ptr,
    stride_qz, stride_qh, stride_qm,
    stride_kz, stride_kh, stride_kn,
    stride_gz, stride_gh, stride_gm,
    stride_Az, stride_Am, stride_An,
    M, N, K, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    b_A = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    for k_offs in range(0, K, BLOCK_K):
        k_mask = k_offs + offs_k < K
        q_block = tl.load(
            q_ptr + offs_m[:, None]*stride_qm + (k_offs + offs_k[None, :])*stride_qh,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0
        )
        k_block = tl.load(
            k_ptr + offs_n[None, :]*stride_km + (k_offs + offs_k[:, None])*stride_kh,
            mask=(offs_n[None, :] < N) & k_mask[:, None],
            other=0.0
        )
        g_block = tl.load(
            g_ptr + offs_m[:, None]*stride_gm + (k_offs + offs_k[None, :])*stride_gh,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0
        )

        prod = tl.dot(q_block, k_block)
        gate = tl.dot(g_block, k_block)
        attn = tl.exp((prod + gate) * scale)
        b_A += attn

    mask_store = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        A_ptr + offs_m[:, None]*stride_Am + offs_n[None, :]*stride_An,
        b_A.to(tl.float32),
        mask=mask_store
    )

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_split(
    q_ptr, k_ptr, g_ptr, A_intra_ptr,
    stride_qz, stride_qh, stride_qm,
    stride_kz, stride_kh, stride_kn,
    stride_gz, stride_gh, stride_gm,
    stride_Aiz, stride_Aim, stride_Ain,
    M, N, K, scale,
    split_start, split_end,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    b_A = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    for k_offs in range(split_start, split_end, BLOCK_K):
        k_mask = k_offs + offs_k < split_end
        q_block = tl.load(
            q_ptr + offs_m[:, None]*stride_qm + (k_offs + offs_k[None, :])*stride_qh,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0
        )
        k_block = tl.load(
            k_ptr + offs_n[None, :]*stride_km + (k_offs + offs_k[:, None])*stride_kh,
            mask=(offs_n[None, :] < N) & k_mask[:, None],
            other=0.0
        )
        g_block = tl.load(
            g_ptr + offs_m[:, None]*stride_gm + (k_offs + offs_k[None, :])*stride_gh,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0
        )

        prod = tl.dot(q_block, k_block)
        gate = tl.dot(g_block, k_block)
        attn = tl.exp((prod + gate) * scale)
        b_A += attn

    mask_store = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        A_intra_ptr + offs_m[:, None]*stride_Aim + offs_n[None, :]*stride_Ain,
        b_A.to(tl.float32),
        mask=mask_store
    )

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge(
    A_intra_ptr, A_ptr,
    stride_Aiz, stride_Aim, stride_Ain,
    stride_Az, stride_Am, stride_An,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    A_intra_val = tl.load(
        A_intra_ptr + offs_m[:, None]*stride_Aim + offs_n[None, :]*stride_Ain,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0
    )
    A_val = tl.load(
        A_ptr + offs_m[:, None]*stride_Am + offs_n[None, :]*stride_An,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0
    )

    merged = A_val + A_intra_val

    mask_store = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        A_ptr + offs_m[:, None]*stride_Am + offs_n[None, :]*stride_An,
        merged,
        mask=mask_store
    )

@triton.jit
def chunk_gla_fwd_kernel_o(
    q_ptr, k_ptr, g_ptr, A_ptr, o_ptr,
    stride_qz, stride_qh, stride_qm,
    stride_kz, stride_kh, stride_kn,
    stride_gz, stride_gh, stride_gm,
    stride_Az, stride_Am, stride_An,
    stride_oz, stride_om, stride_ok,
    M, N, K, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator for final output
    b_O = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    # Dot product across columns of A
    for n_offs in range(0, N, BLOCK_N):
        offs_n = n_offs + tl.arange(0, BLOCK_N)
        A_block = tl.load(
            A_ptr + offs_m[:, None]*stride_Am + offs_n[None, :]*stride_An,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            other=0.0
        )
        k_block = tl.load(
            k_ptr + offs_n[:, None]*stride_km + offs_k[None, :]*stride_kh,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0
        )
        g_block = tl.load(
            g_ptr + offs_m[:, None]*stride_gm + offs_k[None, :]*stride_gh,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        # Weighted sum for output
        b_O += tl.dot(A_block, k_block) * (1 + g_block * scale)

    # Store result
    mask_o = offs_m[:, None] < M
    tl.store(
        o_ptr + offs_m[:, None]*stride
