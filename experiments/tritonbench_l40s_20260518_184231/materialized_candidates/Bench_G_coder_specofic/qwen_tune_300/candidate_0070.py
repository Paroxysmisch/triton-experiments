import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

BT = 16
BK = 4
BV = 8
NC = 8

@triton.jit
def chunk_retention_fwd_kernel_h(
    k,
    v,
    h,
    h0,
    ht,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    boundary_check: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_k = k + i_bh * s_k_h + i_k * BK * s_k_d
    p_v = v + i_bh * s_v_h + i_v * BV * s_v_d
    p_h = h + (i_bh + i_k * NC * i_v) * s_h_h

    if i_k == 0:
        tl.store(p_h, tl.load(h0 + i_bh * K * V + i_v * V * K + i_k * BK * BV), mask=i_v * BV + tl.arange(0, BV) < V)
        p_h += BT * K * V

    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V

    for t in range(0, T, BT):
        mask_bt = t + tl.arange(0, BT) < T

        if boundary_check:
            mask_k = mask_bt[:, None] & mask_bk[None, :] & (t + tl.arange(0, BT)[:, None] >= 0)
            mask_v = mask_bt[:, None] & mask_bv[None, :] & (t + tl.arange(0, BT)[:, None] >= 0)
        else:
            mask_k = mask_bt[:, None] & mask_bk[None, :]
            mask_v = mask_bt[:, None] & mask_bv[None, :]

        p_k_t = p_k + (t + tl.arange(0, BT)[:, None]) * s_k_t
        p_v_t = p_v + (t + tl.arange(0, BT)[:, None]) * s_v_t
        p_h_t = p_h + (t + tl.arange(0, BT)[:, None]) * s_h_t

        mask_kv = mask_k & mask_v

        h_t = tl.load(p_h_t, mask=mask_kv, other=0)
        k_t = tl.load(p_k_t, mask=mask_k, other=0)
        v_t = tl.load(p_v_t, mask=mask_v, other=0)

        h_t += tl.dot(k_t, v_t, allow_tf32=False, out_dtype=tl.float32, out_type=tl.float32)

        tl.store(p_h_t, h_t, mask=mask_kv)

    if i_k == 0:
        tl.store(ht + i_bh * K * V + i_v * V * K + i_k * BK * BV, h_t, mask=i_v * BV + tl.arange(0, BV) < V)

@triton.jit
def chunk_retention_fwd_kernel_o(
    q,
    k,
    v,
    h,
    o,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    block_boundary_check: tl.constexpr,
    chunk_boundary_check: tl.constexpr,
):
    i_v, i_u, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_t = i_u * BT

    p_q = q + i_bh * s_k_h + i_v * BV * s_k_d + i_t * K
    p_k = k + i_bh * s_k_h + i_u * BK * s_k_d + i_t * K
    p_v = v + i_bh * s_v_h + i_u * BV * s_v_d + i_t * V
    p_h = h + i_bh * s_h_h + i_v * BV * s_h_d + i_t * K * V

    mask_kv = (i_t + tl.arange(0, BT)[:, None] < T)[:, None] & ((i_v * BV + tl.arange(0, BV)[None, :]) < V)

    if block_boundary_check:
        mask_q = (i_t + tl.arange(0, BT)[:, None] >= 0)[:, None] & ((i_u * BK + tl.arange(0, BK)[None, :]) < K)
    else:
        mask_q = (i_t + tl.arange(0, BT)[:, None] < T)[:, None] & ((i_u * BK + tl.arange(0, BK)[None, :]) < K)

    mask_k = mask_q & mask_kv
    mask_v = mask_q & mask_kv

    p_o = o + (i_bh + i_v * NC * i_u) * s_k_h + i_t * K

    tl.store(p_o, tl.load(q + i_bh * s_k_h + i_v * BV * s_k_d + i_t * K + (i_u * BK + tl.arange(0, BK)[None, :]) * s_k_t + (i_t + tl.arange(0, BT)[:, None]) * s_k_t, mask=mask_k, other=0), mask=(tl.arange(0, BK)[None, :] < K) & (tl.arange(0, BT)[:, None] < T))

    for _ in range(0, NC - 1):
        p_h += BT * K * V
        p_o += BT * K
        tl.store(p_o, tl.load(p_h, mask=mask_kv, other=0), mask=(tl.arange(0, BT)[:, None] < T) & ((i_v * BV + tl.arange(0, BV)[None, :]) < V))

@triton.jit
def chunk_retention_bwd_kernel_dh(
    q,
    do,
    dh,
    s_k_h,
    s_k_t,
    s_k_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    block_boundary_check: tl.constexpr,
    chunk_boundary_check: tl.constexpr,
):
    i_v, i_u, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_t = (i_u + 1) * BT - 1

    p_q = q + i_bh * s_k_h + i_v * BV * s_k_d + (i_t - tl.arange(0, BT)[:, None]) * K
    p_do = do + i_bh * s_k_h + i_v * BV * s_k_d + (i_t - tl.arange(0, BT)[:, None]) * K
    p_k = k + i_bh * s_k_h + i_u * BK * s_k_d + (i_t - tl.arange(0, BT)[:, None]) * K
    p_h = h + i_bh * s_h_h + i_v * BV * s_h_d + (i_t - tl.arange(0, BT)[:, None]) * K * V

    mask_kv = ((i_t - tl.arange(0, BT)[:, None]) >= 0)[:, None] & ((i_v * BV + tl.arange(0, BV)[None, :]) < V)

    if block_boundary_check:
        mask_q = ((i_t - tl.arange(0, BT)[:, None]) >= 0)[:, None] & ((i_u * BK + tl.arange(0, BK)[None, :]) < K)
    else:
        mask_q = ((i_t - tl.arange(0, BT)[:, None]) >= 0)[:, None] & ((i_t - tl.arange(0, BT)[:, None]) < T)[:, None] & ((i_u * BK + tl.arange(0, BK)[None, :]) < K)

    mask_k = mask_q & mask_kv
    mask_v = mask_q & mask_kv

    for i in range(0, NC):
        p_h -= BT * K * V
        p_do -= BT * K

        tl.store(p_h, tl.load(p_do, mask=mask_kv, other=0), mask=mask_kv)

    tl.store(dh + i_bh * K * V + i_v * V * K + i_t * BK *
