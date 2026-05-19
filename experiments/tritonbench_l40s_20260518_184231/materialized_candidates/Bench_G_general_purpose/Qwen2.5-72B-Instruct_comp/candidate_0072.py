import triton
import triton.language as tl

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q, k, g, A, 
    stride_qb, stride_qh, stride_qk, 
    stride_kb, stride_kh, stride_kk, 
    stride_gb, stride_gh, stride_gk, 
    stride_Ab, stride_Ai, stride_Aj, 
    B, H, N, K, 
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)
    pid_j = tl.program_id(axis=2)

    if pid_i > pid_j:
        return

    b = pid_b
    i = pid_i * BLOCK_SIZE_N
    j = pid_j * BLOCK_SIZE_N

    q_block = tl.load(q + b * stride_qb + i * stride_qh + tl.arange(0, BLOCK_SIZE_K) * stride_qk)
    k_block = tl.load(k + b * stride_kb + j * stride_kh + tl.arange(0, BLOCK_SIZE_K) * stride_kk)
    g_block = tl.load(g + b * stride_gb + i * stride_gh + tl.arange(0, BLOCK_SIZE_K) * stride_gk)

    qk = tl.dot(q_block, k_block, allow_tf32=True)
    qk_scaled = qk * (1.0 / tl.sqrt(K))
    qk_exp = tl.exp(qk_scaled)
    qk_exp_gated = qk_exp * g_block

    b_A = tl.sum(qk_exp_gated, axis=1)

    tl.store(A + b * stride_Ab + i * stride_Ai + j * stride_Aj, b_A)

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q, k, g, A, 
    stride_qb, stride_qh, stride_qk, 
    stride_kb, stride_kh, stride_kk, 
    stride_gb, stride_gh, stride_gk, 
    stride_Ab, stride_Ai, stride_Aj, 
    B, H, N, K, 
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)

    b = pid_b
    i = pid_i * BLOCK_SIZE_N

    q_block = tl.load(q + b * stride_qb + i * stride_qh + tl.arange(0, BLOCK_SIZE_K) * stride_qk)
    k_block = tl.load(k + b * stride_kb + i * stride_kh + tl.arange(0, BLOCK_SIZE_K) * stride_kk)
    g_block = tl.load(g + b * stride_gb + i * stride_gh + tl.arange(0, BLOCK_SIZE_K) * stride_gk)

    qk = tl.dot(q_block, k_block, allow_tf32=True)
    qk_scaled = qk * (1.0 / tl.sqrt(K))
    qk_exp = tl.exp(qk_scaled)
    qk_exp_gated = qk_exp * g_block

    b_A = tl.sum(qk_exp_gated, axis=1)

    tl.store(A + b * stride_Ab + i * stride_Ai + i * stride_Aj, b_A)

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_split(
    q, k, g, A_intra, 
    stride_qb, stride_qh, stride_qk, 
    stride_kb, stride_kh, stride_kk, 
    stride_gb, stride_gh, stride_gk, 
    stride_Ab, stride_Ai, stride_Aj, 
    B, H, N, K, 
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    b = pid_b
    i = pid_i * BLOCK_SIZE_N
    k_start = pid_k * BLOCK_SIZE_K

    q_block = tl.load(q + b * stride_qb + i * stride_qh + k_start * stride_qk + tl.arange(0, BLOCK_SIZE_K))
    k_block = tl.load(k + b * stride_kb + i * stride_kh + k_start * stride_kk + tl.arange(0, BLOCK_SIZE_K))
    g_block = tl.load(g + b * stride_gb + i * stride_gh + k_start * stride_gk + tl.arange(0, BLOCK_SIZE_K))

    qk = tl.dot(q_block, k_block, allow_tf32=True)
    qk_scaled = qk * (1.0 / tl.sqrt(K))
    qk_exp = tl.exp(qk_scaled)
    qk_exp_gated = qk_exp * g_block

    b_A = tl.sum(qk_exp_gated, axis=1)

    tl.store(A_intra + b * stride_Ab + i * stride_Ai + i * stride_Aj + k_start * BLOCK_SIZE_K, b_A)

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge(
    A_intra, A, 
    stride_Ab, stride_Ai, stride_Aj, 
    B, H, N, K, 
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)

    b = pid_b
    i = pid_i * BLOCK_SIZE_N

    A_intra_block = tl.load(A_intra + b * stride_Ab + i * stride_Ai + i * stride_Aj + tl.arange(0, K))
    A_block = tl.sum(A_intra_block, axis=1)

    tl.store(A + b * stride_Ab + i * stride_Ai + i * stride_Aj, A_block)

@triton.jit
def chunk_gla_fwd_kernel_o(
    A, v, o, 
    stride_Ab, stride_Ai, stride_Aj, 
    stride_vb, stride_vh, stride_vk, 
    stride_ob, stride_oh, stride_ok, 
    B, H, N, K, 
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_i = tl.program_id(axis=1)

    b = pid_b
    i = pid_i * BLOCK_SIZE_N

    A_block = tl.load(A + b * stride_Ab + i * stride_Ai + tl.arange(0, N) * stride_Aj)
    v_block = tl.load(v + b * stride_vb + i * stride_vh + tl.arange(0, K) * stride_vk)

    o_block = tl.dot(A_block, v_block, allow_tf32=True)

    tl.store(o + b * stride_ob + i * stride_oh + tl.arange(0, K) * stride_ok, o_block)
