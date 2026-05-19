import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    Q_PTR, K_PTR, V_PTR, H_PTR, G_PTR,
    DO_PTR, DH_PTR,
    DQ_PTR, DK_PTR, DG_PTR,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    stride_qb, stride_qh, stride_qt,
    stride_kb, stride_kh, stride_kt,
    stride_vb, stride_vh, stride_vt,
    stride_hb, stride_hh, stride_ht,
    stride_gb, stride_gh, stride_gt,
    stride_dob, stride_doh, stride_dot,
    stride_dhb, stride_dhh, stride_dht,
    stride_dqb, stride_dqh, stride_dqt,
    stride_dkb, stride_dkh, stride_dkt,
    stride_dgb, stride_dgh, stride_dgt,
):
    pid_k = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_bh = tl.program_id(2)

    bh = pid_bh
    if bh >= BT:
        return

    b = bh // (BT // (BT // 1))
    h = bh % (BT // (BT // 1))

    k_offset = pid_k * BK
    t_offset = pid_t * BK

    b_dq = tl.zeros([BK], tl.float32)
    b_dk = tl.zeros([BK], tl.float32)
    b_dg = tl.zeros([BK], tl.float32)

    for v_i in range(0, BV):
        k_valid = (k_offset + tl.arange(0, BK)) < BK
        t_valid = (t_offset + tl.arange(0, BK)) < BT

        q_ptrs = Q_PTR + b * stride_qb + h * stride_qh + (t_offset + tl.arange(0, BK)) * stride_qt
        k_ptrs = K_PTR + b * stride_kb + h * stride_kh + (k_offset + tl.arange(0, BK)) * stride_kt
        v_ptrs = V_PTR + b * stride_vb + h * stride_vh + v_i * stride_vt
        h_ptrs = H_PTR + b * stride_hb + h * stride_hh + v_i * stride_ht
        g_ptrs = G_PTR + b * stride_gb + h * stride_gh + (t_offset + tl.arange(0, BK)) * stride_gt
        do_ptrs = DO_PTR + b * stride_dob + h * stride_doh + (t_offset + tl.arange(0, BK)) * stride_dot
        dh_ptrs = DH_PTR + b * stride_dhb + h * stride_dhh + v_i * stride_dht

        mask_load_q = k_valid & t_valid
        mask_load_k = k_valid
        q_val = tl.where(mask_load_q, tl.load(q_ptrs, mask=mask_load_q, other=0.), 0.)
        k_val = tl.where(mask_load_k, tl.load(k_ptrs, mask=mask_load_k, other=0.), 0.)
        v_val = tl.load(v_ptrs) if v_i < BV else 0.
        h_val = tl.load(h_ptrs) if v_i < BV else 0.
        do_val = tl.where(mask_load_q, tl.load(do_ptrs, mask=mask_load_q, other=0.), 0.)
        dh_val = tl.load(dh_ptrs) if v_i < BV else 0.
        g_val = tl.where(mask_load_q, tl.load(g_ptrs, mask=mask_load_q, other=0.), 0.)

        dq_part = do_val * g_val * k_val  # example partial
        dk_part = do_val * g_val * q_val
        dg_part = do_val * (q_val * k_val)

        b_dq += dq_part
        b_dk += dk_part
        b_dg += dg_part

    dq_out_ptrs = DQ_PTR + b * stride_dqb + h * stride_dqh + (t_offset + tl.arange(0, BK)) * stride_dqt
    dk_out_ptrs = DK_PTR + b * stride_dkb + h * stride_dkh + (k_offset + tl.arange(0, BK)) * stride_dkt
    dg_out_ptrs = DG_PTR + b * stride_dgb + h * stride_dgh + (t_offset + tl.arange(0, BK)) * stride_dgt

    mask_store_q = (t_offset + tl.arange(0, BK)) < BT
    mask_store_k = (k_offset + tl.arange(0, BK)) < BK

    tl.store(dq_out_ptrs, b_dq, mask=mask_store_q)
    tl.store(dk_out_ptrs, b_dk, mask=mask_store_k)
    tl.store(dg_out_ptrs, b_dg, mask=mask_store_q)


def chunk_bwd_dqkg_fn(q, k, v, h, g, do, dh):
    B, H, T = q.shape
    BK = k.shape[-1]
    BT = T
    BV = v.shape[-1]

    dq = tl.zeros_like(q)
    dk = tl.zeros_like(k)
    dg = tl.zeros_like(q)

    grid = (BK // BK, T // BK, B * H)

    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q, k, v, h, g, do, dh, dq, dk, dg,
        BT, BK, BV,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        g.stride(0), g.stride(1), g.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        dh.stride(0), dh.stride(1), dh.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dg.stride(0), dg.stride(1), dg.stride(2),
    )
    return dq, dk, dg
