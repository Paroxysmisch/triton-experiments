import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['BT', 'BK', 'BV']
)
@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, g_ptr, o_ptr,
    s_q_h, s_q_t, s_q_d,
    s_k_h, s_k_t, s_k_d,
    s_v_h, s_v_t, s_v_d,
    s_h_h, s_h_t, s_h_d,
    s_g_h, s_g_t, s_g_d,
    s_o_h, s_o_t, s_o_d,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    scale: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Compute program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute the block bounds
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize pointers to Q, K, V, H, G
    q_ptrs = q_ptr + (offs_m[:, None] * s_q_t + offs_k[None, :] * s_q_d + tl.arange(0, BT) * s_q_h)
    k_ptrs = k_ptr + (offs_k[:, None] * s_k_d + offs_n[None, :] * s_k_t + tl.arange(0, BT) * s_k_h)
    v_ptrs = v_ptr + (offs_k[:, None] * s_v_d + offs_n[None, :] * s_v_t + tl.arange(0, BT) * s_v_h)
    h_ptrs = h_ptr + (offs_m[:, None] * s_h_t + offs_k[None, :] * s_h_d + tl.arange(0, BT) * s_h_h)
    g_ptrs = g_ptr + (offs_m[:, None] * s_g_t + offs_k[None, :] * s_g_d + tl.arange(0, BT) * s_g_h)

    # Load Q, K, V, H, G
    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)
    h = tl.load(h_ptrs)
    g = tl.load(g_ptrs)

    # Compute QK^T
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        qk = tl.dot(q, k, allow_tf32=True)
        acc += qk

    # Apply scale
    acc *= scale

    # Compute softmax
    m = tl.max(acc, 1)
    p = tl.exp(acc - m[:, None])
    p_sum = tl.sum(p, 1)
    p = p / p_sum[:, None]

    # Compute output
    o = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        o += tl.dot(p, v, allow_tf32=True)

    # Store the output
    o_ptrs = o_ptr + (offs_m[:, None] * s_o_t + offs_n[None, :] * s_o_d + tl.arange(0, BT) * s_o_h)
    tl.store(o_ptrs, o)

def chunk_fwd_o_fn(q, k, v, h, g, o, scale, BT, BK, BV):
    # Compute grid dimensions
    M, N, K = q.shape[1], k.shape[1], q.shape[2]
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)

    # Compute strides
    s_q_h, s_q_t, s_q_d = q.stride(0), q.stride(1), q.stride(2)
    s_k_h, s_k_t, s_k_d = k.stride(0), k.stride(1), k.stride(2)
    s_v_h, s_v_t, s_v_d = v.stride(0), v.stride(1), v.stride(2)
    s_h_h, s_h_t, s_h_d = h.stride(0), h.stride(1), h.stride(2)
    s_g_h, s_g_t, s_g_d = g.stride(0), g.stride(1), g.stride(2)
    s_o_h, s_o_t, s_o_d = o.stride(0), o.stride(1), o.stride(2)

    # Launch the kernel
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, g, o,
        s_q_h, s_q_t, s_q_d,
        s_k_h, s_k_t, s_k_d,
        s_v_h, s_v_t, s_v_d,
        s_h_h, s_h_t, s_h_d,
        s_g_h, s_g_t, s_g_d,
        s_o_h, s_o_t, s_o_d,
        BT, BK, BV,
        scale,
        M, N, K,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32
    )
