@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    q, k, v, h, g, do, dh, dq, dk, dg,
    DEC_DIV: tl.constexpr,
    STRIDE_QH: tl.constexpr, STRIDE_KH: tl.constexpr, STRIDE_VG: tl.constexpr,
    STRIDE_DQ: tl.constexpr, STRIDE_DK: tl.constexpr, STRIDE_DG: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    num_stages: tl.constexpr, num_warps: tl.constexpr
):
    i_k, i_n, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % num_warps

    qk_mask = (1 - tl.math.pow(2, -BT)) * 40
    exp_list = tl.math.exp2((qk_mask[i_h] - qk_mask[:BT]) / DEC_DIV)

    b_dq = tl.zeros((BK, BV), dtype=tl.float32)
    b_dk = tl.zeros((BK, BV), dtype=tl.float32)
    b_dg = tl.zeros((BK, BV), dtype=tl.float32)

    for i in range(0, tl.cdiv(BV, BT)):
        p_h = tl.make_block_ptr(h + i_bh * STRIDE_QH, (BK, BV), (BV, 1), (i_n * BK, i * BT), (BK, BT), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * STRIDE_KH, (BK, BV), (BV, 1), (i_n * BK, i * BT), (BK, BT), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * STRIDE_VG, (BV, BK), (BK, 1), (i * BT, i_n * BK), (BT, BK), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * STRIDE_DQ, (BV, BK), (BK, 1), (i * BT, i_n * BK), (BT, BK), (1, 0))

        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        b_doh = tl.dot(b_do, b_h, allow_tf32=False)
        b_dgh = tl.dot(b_dh.to(b_do.dtype), b_h, allow_tf32=False)
        b_dqh = tl.dot(b_doh * exp_list, b_h, allow_tf32=False)
        b_dgh = b_dgh * exp_list
        b_dkh = tl.dot((b_doh.to(b_dh.dtype) + tl.dot(b_g, b_dh, allow_tf32=False)) * exp_list, b_h, allow_tf32=False)

        b_dq += b_dqh
        b_dk += b_dkh
        b_dg += b_dgh

        p_dk = tl.make_block_ptr(dk + i_bh * STRIDE_DK, (BK, BV), (BV, 1), (i_n * BK, i * BT), (BK, BT), (1, 0))
        p_dg = tl.make_block_ptr(dg + i_bh * STRIDE_DG, (BK, BV), (BV, 1), (i_n * BK, i * BT), (BK, BT), (1, 0))

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
