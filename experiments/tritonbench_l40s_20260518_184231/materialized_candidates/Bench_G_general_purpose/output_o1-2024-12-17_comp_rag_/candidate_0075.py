import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=8),
    ],
    key=['q_ptr', 'k_ptr', 'v_ptr', 'h_ptr', 'g_ptr'],
)
@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, g_ptr, o_ptr,
    s_q_bs, s_q_h, s_q_t, s_q_d,
    s_k_bs, s_k_h, s_k_t, s_k_d,
    s_v_bs, s_v_h, s_v_t, s_v_d,
    s_h_bs, s_h_h, s_h_t, s_h_d,
    s_g_bs, s_g_h, s_g_t,
    s_o_bs, s_o_h, s_o_t, s_o_d,
    scale, BT, BK, BV,
    BLOCK_SIZE: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for query, key, value, etc.
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Pointers for Q, K, H, G
    p_q = q_ptr + offs_m[:, None] * s_q_t + offs_n[None, :] * s_q_d
    p_h = h_ptr + offs_m[:, None] * s_h_t + offs_n[None, :] * s_h_d
    p_k = k_ptr + offs_m[:, None] * s_k_t + offs_n[None, :] * s_k_d
    p_g = g_ptr + offs_m * s_g_t

    # Load Q, K, H, G
    q_matrix = tl.load(p_q, mask=(offs_m[:, None] < BT) & (offs_n[None, :] < BK), other=0.0)
    k_matrix = tl.load(p_k, mask=(offs_m[:, None] < BT) & (offs_n[None, :] < BK), other=0.0)
    h_matrix = tl.load(p_h, mask=(offs_m[:, None] < BT) & (offs_n[None, :] < BK), other=0.0)
    g_vector = tl.load(p_g, mask=(offs_m < BT), other=0.0)

    # Compute partial outputs
    b_o = tl.dot(q_matrix, k_matrix, out_dtype=tl.float32) * scale
    b_s = tl.exp(b_o) * g_vector[:, None] * h_matrix

    # Mask condition (example usage)
    m_s = b_s > 0  # simplistic condition
    b_s = tl.where(m_s, b_s, 0.0)

    # Pointers for output O
    p_o = o_ptr + offs_m[:, None] * s_o_t + offs_n[None, :] * s_o_d

    # Store result
    tl.store(
        p_o,
        b_s.to(tl.float16),
        mask=(offs_m[:, None] < BT) & (offs_n[None, :] < BK)
    )

def chunk_fwd_o_fn(
    q, k, v, h, g, o,
    scale: float,
    BT: int,
    BK: int,
    BV: int
):
    import math

    # Prepare grid dimensions
    grid_m = math.ceil(BT / 64)
    grid_n = math.ceil(BK / 64)
    grid = (grid_m, grid_n)

    # Launch kernel
    chunk_simple_gla_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        h.stride(0), h.stride(1), h.stride(2), h.stride(3),
        g.stride(0), g.stride(1), g.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        scale, BT, BK, BV,
    )
