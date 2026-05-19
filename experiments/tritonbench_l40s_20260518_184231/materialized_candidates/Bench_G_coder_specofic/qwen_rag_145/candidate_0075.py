import triton.language as tl
import triton

@triton.autotune
def chunk_simple_gla_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, g_ptr, o_ptr,
    s_q, s_k_h, s_k_t, s_v_h, s_v_t, s_h, s_g,
    BT, BK, BV,
    scale, exp_scale,
    p_q, p_k, p_v, p_h, p_g,
    m_s,
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)
    row_offset = pid_row * BK
    col_offset = pid_col * BV

    # Load
    q_local = tl.load(q_ptr + row_offset, s_q)[:, pid_col : pid_col + BV]
    k_local = tl.load(k_ptr + col_offset, s_k_h)[:BT, :]
    v_local = tl.load(v_ptr + col_offset, s_v_h)[:BT, :]
    h_local = tl.load(h_ptr + col_offset, s_h)[:BT, :]
    g_local = tl.load(g_ptr + row_offset, s_g)[:, pid_col : pid_col + BV]

    # Compute
    b_o = q_local @ k_local * scale
    b_s = v_local @ h_local * exp_scale
    masked_s = tl.where(m_s, b_s, 0.)

    # Store
    o_local = b_o + masked_s + g_local
    tl.store(o_ptr + row_offset, o_local, s_g)
