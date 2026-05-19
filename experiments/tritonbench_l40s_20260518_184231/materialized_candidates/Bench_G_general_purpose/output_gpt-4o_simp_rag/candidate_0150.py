import triton
import triton.language as tl

@triton.jit
def chunk_retention_fwd_kernel_h(
    k, v, h, final_state,
    T: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_k = k + i_bh * T * D + i_t * BT * D + o_d
    p_v = v + i_bh * T * D + i_t * BT * D + o_d
    p_h = h + i_bh * T * D + i_t * BT * D + o_d

    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE and i_t == 0:
        b_h += tl.load(final_state + i_bh * D + o_d, mask=mask, other=0).to(tl.float32)

    for i in range(0, BT):
        mask_t = mask & ((i_t * BT + i) < T)
        b_k = tl.load(p_k, mask=mask_t, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_t, other=0).to(tl.float32)
        b_h = b_h + b_k * b_v
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_t)

        p_k += D
        p_v += D
        p_h += D

@triton.jit
def chunk_retention_fwd_kernel_o(
    q, k, v, h, o,
    T: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr
):
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    for i_t in range(0, tl.cdiv(T, BT)):
        p_q = q + i_bh * T * D + i_t * BT * D + o_d
        p_k = k + i_bh * T * D + i_t * BT * D + o_d
        p_v = v + i_bh * T * D + i_t * BT * D + o_d
        p_h = h + i_bh * T * D + i_t * BT * D + o_d
        p_o = o + i_bh * T * D + i_t * BT * D + o_d

        b_q = tl.load(p_q, mask=mask, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask, other=0).to(tl.float32)
        b_h = tl.load(p_h, mask=mask, other=0).to(tl.float32)

        b_o = b_q * b_h + b_k * b_v
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask)

@triton.jit
def chunk_retention_bwd_kernel_dh(
    dh, do, T: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_dh = dh + i_bh * T * D + i_t * BT * D + o_d
    p_do = do + i_bh * T * D + i_t * BT * D + o_d

    b_dh = tl.zeros([BD], dtype=tl.float32)
    for i in range(0, BT):
        mask_t = mask & ((i_t * BT + i) < T)
        b_do = tl.load(p_do, mask=mask_t, other=0).to(tl.float32)
        b_dh = b_dh + b_do
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), mask=mask_t)

        p_dh += D
        p_do += D

@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    q, k, v, h, dq, dk, dv, dh, do,
    T: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr
):
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    for i_t in range(0, tl.cdiv(T, BT)):
        p_q = q + i_bh * T * D + i_t * BT * D + o_d
        p_k = k + i_bh * T * D + i_t * BT * D + o_d
        p_v = v + i_bh * T * D + i_t * BT * D + o_d
        p_h = h + i_bh * T * D + i_t * BT * D + o_d
        p_dq = dq + i_bh * T * D + i_t * BT * D + o_d
        p_dk = dk + i_bh * T * D + i_t * BT * D + o_d
        p_dv = dv + i_bh * T * D + i_t * BT * D + o_d
        p_dh = dh + i_bh * T * D + i_t * BT * D + o_d
        p_do = do + i_bh * T * D + i_t * BT * D + o_d

        b_q = tl.load(p_q, mask=mask, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask, other=0).to(tl.float32)
        b_h = tl.load(p_h, mask=mask, other=0).to(tl.float32)
        b_dh = tl.load(p_dh, mask=mask, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)

        b_dq = b_dh * b_q + b_do
        b_dk = b_dh * b_k
        b_dv = b_dh * b_v

        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask)
