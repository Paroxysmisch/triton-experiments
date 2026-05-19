import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd.function import Function

@triton.jit
def chunk_linear_attn_fwd_kernel_h(
    k,
    v,
    h0,
    h,
    chunk_size,
    K,
    V,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_v, i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3)
    p_k = k + i_bh * K * V + i_c * BT * BK + tl.arange(0, BK) + tl.arange(0, BT)[:, None] * K
    p_v = v + i_bh * V * chunk_size + i_c * BT * BV + tl.arange(0, BV) + tl.arange(0, BT)[:, None] * V
    p_h = h + i_bh * chunk_size * V + i_c * BT * BV + tl.arange(0, BV) + tl.arange(0, BT)[:, None] * V
    mask_bt = tl.arange(0, BT) < chunk_size
    mask_kv = (tl.arange(0, BK) < K) & (tl.arange(0, BV) < V)
    mask_bh = i_bh < K // BT
    h_i = tl.zeros([BT, BV], dtype=tl.float32)
    if i_c * BT * BK * BV < K * V:
        if h0 is not None:
            h_i = tl.load(h0 + i_bh * chunk_size * V + (i_c * BT * BV + tl.arange(0, BV) + tl.arange(0, BT)[:, None] * V), mask=(i_bh * chunk_size * V + i_c * BT * BV + tl.arange(0, BV) + tl.arange(0, BT)[:, None] * V < chunk_size * V) & mask_bh, other=0).to(tl.float32)
        for i in range(0, tl.cdiv(K, BK)):
            mask = mask_bt[:, None] & mask_kv[None, :] & (i_k == i)
            a = tl.load(p_k, mask=mask, other=0).to(tl.float32)
            b = tl.load(p_v, mask=mask, other=0).to(tl.float32)
            h_i += tl.dot(a, b, allow_tf32=False)
        tl.store(p_h, h_i.to(p_h.dtype.element_ty), mask=mask_bt[:, None] & mask_kv[None, :])

@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    q,
    k,
    v,
    h,
    o,
    chunk_size,
    K,
    V,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * K * V + i_v * BT * BV + tl.arange(0, BV) + tl.arange(0, BT)[:, None] * V
    p_k = k + i_bh * K * V + i_k * BT * BK + tl.arange(0, BK) + tl.arange(0, BT)[:, None] * K
    p_v = v + i_bh * V * chunk_size + i_v * BT * BV + tl.arange(0, BV) + tl.arange(0, BT)[:, None] * V
    p_h = h + i_bh * chunk_size * V + i_k * BT * BV + tl.arange(0, BV) + tl.arange(0, BT)[:, None] * V
    p_o = o + i_bh * K * V + i_v * BT * BV + tl.arange(0, BV) + tl.arange(0, BT)[:, None] * V
    mask_bt = tl.arange(0, BT) < chunk_size
    mask_k = tl.arange(0, BK) < K
    mask_v = tl.arange(0, BV) < V
    mask_bh = i_bh < K // BT
    for i in range(0, tl.cdiv(V, BV)):
        mask_p_k = mask_bh & (i_k * BT * BK + tl.arange(0, BK) + tl.arange(0, BT)[:, None] * K < K)
        mask_p_v = mask_bh & (i_v * BT * BV + tl.arange(0, BV) + tl.arange(0, BT)[:, None] * V < V)
        q_i = tl.load(p_q, mask=mask_p_v, other=0).to(tl.float32)
        tl.store(p_o, q_i, mask=mask_p_v)
        p_h_i = p_h + (i_k * BT * BV + tl.arange(0, BV) + tl.arange(0, BT)[:, None] * V)
        o_i = tl.zeros([BT, BV], dtype=tl.float32)
        for j in range(0, tl.cdiv(K, BK)):
            mask_h = mask_bt[:, None] & (i_k == j)
            h_j = tl.load(p_h_i, mask=mask_h, other=0).to(tl.float32)
            o_i += h_j
            p_h_i += BV
        p_k_i = p_k + (i_k * BT * BK + tl.arange(0, BK) + tl.arange(0, BT)[:, None] * K)
        for j in range(0, tl.cdiv(K, BK)):
            mask = mask_bt[:, None] & mask_k[None, :] & (i_k == j)
            a = tl.load(p_k_i, mask=mask, other=0).to(tl.float32)
            b = tl.load(p_v, mask=mask, other=0).to(tl.float32)
            o_i += tl.dot(a, b, allow_tf32=False)
            p_k_i += BK
        tl.store(p_o, o_i.to(p_o.dtype.element_ty), mask=mask_p_v)

@triton.jit
def chunk_linear_attn_bwd_kernel_dh(
    q,
    k,
    v,
    do,
    dq,
    dk,
    dv,
    h,
    chunk_size,
    K,
    V,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * K * V + i_v * BT * BV + t
