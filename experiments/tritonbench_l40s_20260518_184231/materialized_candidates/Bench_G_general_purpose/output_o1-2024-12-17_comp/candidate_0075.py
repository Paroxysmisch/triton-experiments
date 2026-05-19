import triton
import triton.language as tl

# -----------------------------------------------------------------------------------
# Kernel: chunk_simple_gla_fwd_kernel_o
# -----------------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    ],
    key=["BK", "BV"],
)
@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, g_ptr, o_ptr, m_s_ptr,
    s_q_h, s_q_t, s_q_c,
    s_k_h, s_k_t, s_k_c,
    s_v_h, s_v_t, s_v_c,
    s_h_h, s_h_t, s_h_c,
    s_g_h, s_g_t, s_g_c,
    s_o_h, s_o_t, s_o_c,
    scale,
    BT, BK, BV,
    BLOCK_SIZE: tl.constexpr
):
    # Program ids to partition the problem space
    pid_b = tl.program_id(0)  # batch dimension or partial chunk ID
    pid_t = tl.program_id(1)  # token dimension chunk
    base_t = pid_t * BT

    # Offsets for Q, K, V, H, G, M_S, O
    q_offset = pid_b * s_q_h + base_t * s_q_t
    k_offset = pid_b * s_k_h
    v_offset = pid_b * s_v_h
    h_offset = pid_b * s_h_h + base_t * s_h_t
    g_offset = pid_b * s_g_h
    ms_offset = base_t  # simplistic assumption for row offset in mask
    o_offset = pid_b * s_o_h + base_t * s_o_t

    # Load Q and H sub-blocks for the tokens in [base_t : base_t + BT]
    rid = tl.arange(0, BLOCK_SIZE)
    # Some threads might go out of range if BT < BLOCK_SIZE, guarded by mask
    tmask = rid < BT

    # Pointers to Q, H
    p_q = q_ptr + q_offset + rid * s_q_t
    p_h = h_ptr + h_offset + rid * s_h_t

    # Initialize accumulators for output
    # b_o for partial dot products with V, b_s for accumulated exponentials
    b_o = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    b_s = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over K dimension in steps of BK
    # chunked calculation
    for chunk_id in range(0, BK):  # naive loop, can be optimized
        # Offsets for K, V, G for each chunk
        k_ch_offset = k_offset + chunk_id * s_k_t * BK
        v_ch_offset = v_offset + chunk_id * s_v_t * BK
        g_ch_offset = g_offset + chunk_id * s_g_t * BK

        # Load K-chunk
        # We'll do a small partial sum per thread
        q_val = tl.load(p_q + 0 * s_q_c, mask=tmask, other=0.0)
        k_val = tl.load(k_ptr + k_ch_offset + rid * s_k_t, mask=tmask, other=0.0)

        # Dot product Q*K
        dot_qk = q_val * k_val
        dot_val = tl.sum(dot_qk, axis=0)
        dot_scaled = dot_val * scale  # multiply by scale

        # Softmax with partial exponentials
        exp_val = tl.exp(dot_scaled)
        # Get mask if needed
        if m_s_ptr is not None:
            m_s_val = tl.load(m_s_ptr + ms_offset + rid, mask=tmask, other=0.0)
            exp_val = tl.where(m_s_val > 0.0, exp_val, 0.0)

        # Accumulate partial sums
        b_s += exp_val

        # Multiply by V: each thread loads the corresponding V
        v_val = tl.load(v_ptr + v_ch_offset + rid * s_v_t, mask=tmask, other=0.0)
        b_o += v_val * exp_val

        # Example usage of H or G (dummy usage here)
        h_val = tl.load(p_h + 0 * s_h_c, mask=tmask, other=0.0)
        g_val = tl.load(g_ptr + g_ch_offset + rid * s_g_t, mask=tmask, other=1.0)
        # Possibly incorporate gating
        b_o += h_val * g_val * exp_val

    # Normalize b_o by b_s
    out_val = b_o / (b_s + 1e-6)

    # Store to O
    tl.store(o_ptr + o_offset + rid * s_o_t, out_val, mask=tmask)


# -----------------------------------------------------------------------------------
# Wrapper: chunk_fwd_o_fn
# -----------------------------------------------------------------------------------
def chunk_fwd_o_fn(q, k, v, h, g, o, m_s,
                   scale, BT, BK, BV,
                   s_q, s_k, s_v, s_h, s_g, s_o):
    """
    High-level orchestration to run the kernel.
    s_q, s_k, s_v, s_h, s_g, s_o: dictionary of strides for q, k, v, h, g, o
    BT, BK, BV: chunk sizes
    """
    # Grid dimensions
    # Let's assume each chunk is one in the batch dimension, but typically
    # we partition across batch as well. We'll use a simplistic setting:
    batch_size = q.shape[0]  # or deduce from shapes
    # We create a grid: (batch_size, num_blocks_in_token_dim)
    T = q.shape[1]  # tokens
    grid_t = (T + BT - 1) // BT
    grid = (batch_size, grid_t)

    # Launch kernel
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, g, o, m_s,
        s_q["h"], s_q["t"], s_q["c"],
        s_k["h"], s_k["t"], s_k["c"],
        s_v["h"], s_v["t"], s_v["c"],
        s_h["h"], s_h["t"], s_h["c"],
        s_g["h"], s_g["t"], s_g["c"],
        s_o["h"], s_o["t"], s_o["c"],
        scale,
        BT, BK, BV
    )
