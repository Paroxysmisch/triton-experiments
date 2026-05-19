import triton
import triton.language as tl

@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=2),
])
@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, g_ptr, o_ptr,
    s_q_h, s_q_t, s_k_h, s_k_t, s_v_h, s_v_t,
    s_h_h, s_h_t, s_g_h, s_g_t, s_o_h, s_o_t,
    scale, BT, BK, BV,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to blocks
    p_q = q_ptr + offs_m[:, None] * s_q_h + offs_k[None, :] * s_q_t
    p_k = k_ptr + offs_k[:, None] * s_k_h + offs_n[None, :] * s_k_t
    p_v = v_ptr + offs_k[:, None] * s_v_h + offs_n[None, :] * s_v_t
    p_h = h_ptr + offs_m[:, None] * s_h_h + offs_k[None, :] * s_h_t
    p_g = g_ptr + offs_m[:, None] * s_g_h + offs_k[None, :] * s_g_t

    # Load data into registers
    q = tl.load(p_q)
    k = tl.load(p_k)
    v = tl.load(p_v)
    h = tl.load(p_h)
    g = tl.load(p_g)

    # Compute partial outputs
    b_o = tl.dot(q, k) * scale
    b_s = tl.dot(h, g)

    # Apply mask and compute exponential
    m_s = tl.exp(b_o) * b_s

    # Store result
    p_o = o_ptr + offs_m[:, None] * s_o_h + offs_n[None, :] * s_o_t
    tl.store(p_o, m_s)


def chunk_fwd_o_fn(q, k, v, h, g, o, scale, BT, BK, BV):
    # Extract shapes
    H, T = q.shape
    _, K = k.shape
    _, V = v.shape

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Calculate grid dimensions
    grid = (triton.cdiv(H, BLOCK_SIZE_M), triton.cdiv(T, BLOCK_SIZE_N))

    # Launch the kernel
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, g, o,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        h.stride(0), h.stride(1),
        g.stride(0), g.stride(1),
        o.stride(0), o.stride(1),
        scale, BT, BK, BV,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
