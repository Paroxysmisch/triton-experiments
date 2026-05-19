import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def fwd_decay_cumsum(
    g, g_o, inv_ln2,
    BT: tl.constexpr, BK: tl.constexpr, DK: tl.constexpr,
    BK_MIN: tl.constexpr,
    DK_NEXT: tl.constexpr,
    BDIV_BK: tl.constexpr, TDIV_BK: tl.constexpr,
    BK_NEXT: tl.constexpr,
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k_next = tl.minimum(i_k * BK + BK, DK)

    p_g = g + (i_bh * TDIV_BK + i_c * TDIV_BK * DK + i_k * BK + tl.arange(0, BK))
    p_go = g_o + (i_bh * TDIV_BK + i_c * TDIV_BK * DK + i_k * BK + tl.arange(0, BK))

    cum_decay = tl.zeros([BK], dtype=tl.float32)
    mask = (tl.arange(0, BK) + i_k * BK) < DK

    for i in range(BT):
        vec = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        cum_decay += vec * inv_ln2
        vec_out = cum_decay.to(p_go.dtype.element_ty)
        tl.store(p_go, vec_out, mask=mask)
        p_g += BK
        p_go += BK

@triton.jit
def prepare_qg_kg(
    q, k, g, qg, kg,
    BK: tl.constexpr, DK: tl.constexpr, DG: tl.constexpr,
    BT: tl.constexpr, BK_MIN: tl.constexpr,
    DK_NEXT: tl.constexpr,
    BDIV_BK: tl.constexpr, TDIV_BK: tl.constexpr,
    BK_NEXT: tl.constexpr,
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + (i_bh * TDIV_BK + i_c * TDIV_BK * DK + i_k * BK + tl.arange(0, BK))
    p_k = k + (i_bh * TDIV_BK + i_c * TDIV_BK * DK + i_k * BK + tl.arange(0, BK))
    p_g = g + (i_bh * TDIV_BK + i_c * TDIV_BK * DG + i_k * BK + tl.arange(0, BK))
    p_qg = qg + (i_bh * TDIV_BK * DK_NEXT + i_c * TDIV_BK * DK + i_k * BK + tl.arange(0, BK))
    p_kg = kg + (i_bh * TDIV_BK * DK_NEXT + i_c * TDIV_BK * DK + i_k * BK + tl.arange(0, BK))
    mask = (tl.arange(0, BK) + i_k * BK) < DK
    cum_decay = tl.zeros([BK], dtype=tl.float32)
    for i in range(BT):
        vec_q = tl.load(p_q, mask=mask, other=0).to(tl.float32)
        vec_k = tl.load(p_k, mask=mask, other=0).to(tl.float32)
        vec_g = tl.load(p_g, mask=mask, other=0)
        cum_decay += vec_g
        vec_q *= tl.exp(cum_decay) * vec_g
        vec_k *= tl.exp(cum_decay) * vec_g
        tl.store(p_qg, vec_q.to(p_qg.dtype.element_ty), mask=mask)
        tl.store(p_kg, vec_k.to(p_kg.dtype.element_ty), mask=mask)
        p_q += BK
        p_k += BK
        p_g += BK
        p_qg += BK
        p_kg += BK

@triton.jit
def bwd_decay_global_cumsum(
    dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg,
    BT: tl.constexpr, BK: tl.constexpr, DK: tl.constexpr,
    DG: tl.constexpr,
    BK_MIN: tl.constexpr,
    DK_NEXT: tl.constexpr,
    BDIV_BK: tl.constexpr, TDIV_BK: tl.constexpr,
    BK_NEXT: tl.constexpr,
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_k_next = tl.minimum(i_k * BK + BK, DK)
    inv_tie_strength = tl.load(
        g + DK + DG * BDIV_BK + i_bh * TDIV_BK + i_c * TDIV_BK * DG + i_k * BK + tl.arange(0, BK),
        mask=(tl.arange(0, BK) + i_k * BK) < DK,
        other=0,
    )

    p_dq = dq_inner + (i_bh * TDIV_BK * DK_NEXT + i_c * TDIV_BK * DK + i_k * BK + tl.arange(0, BK))
    p_dg = dg + (i_bh * TDIV_BK + i_c * TDIV_BK * DG + i_k * BK + tl.arange(0, BK))
    p_dk = dk_inner + (i_bh * TDIV_BK * DK_NEXT + i_c * TDIV_BK * DK + i_k * BK + tl.arange(0, BK))
    mask = (tl.arange(0, BK) + i_k * BK) < DK

    # Use an accumulator to handle the sum in the quotient rule.
    dq_accum = tl.zeros([BK], dtype=tl.float32)
    dk_accum = tl.zeros([BK], dtype=tl.float32)

    for j in range(BT - 1, -1, -1):
        vec_dq = tl.load(p_dq, mask=mask, other=0).to(tl.float32)
        vec_dg = tl.load(p_dg, mask=mask, other=0).to(tl.float32)
        vec_k = tl.load(
            k + (i_bh * TDIV_BK + i_c * TDIV_BK * DK + i_k * BK + tl.arange(0, BK)),
            mask=mask,
            other=0,
        ).to(tl.float32)
        vec_q = tl.load(
            q + (i_bh * TDIV_BK + i_c * TDIV_BK * DK + i_k * BK + tl.arange(0, BK)),
            mask=mask,
            other=0,
        ).to(tl.float32)
        vec_g = tl.load(
            g + (i_bh * TDIV_BK + i_c * TDIV_BK * DG + i_k * BK + tl.arange(0, BK)),
            mask=mask,
            other=0,
        ).to(tl.float32)

        # Update the sum in the quotient rule accumulator.
        dq_accum += vec_dq
        dk_accum += vec_dk

        # Compute the partial derivatives.
        vec_q *= inv_tie_strength
        vec_k *= inv_tie_strength
        vec_dq *= vec_g
        vec_dg *= vec_q
        vec_dk *= vec_g

        # vec_dq += vec_dg
        # vec_dk += vec_dg

        vec_dq -= vec_k * tl.sum(vec_k * dq_accum, axis=0) / tl.sum(vec_k * vec_k, axis=0)
        vec_dk -= vec_q * tl.sum(vec_q * dk_accum, axis=0) / tl.sum(vec_q * vec_q, axis=0)

        tl.store(p_dq, vec_dq.to(p_dq.dtype.element_ty), mask=mask)
        tl.store(dg, vec_dg.to(dg.dtype.element_ty), mask=mask)
        tl.store(p_dk, vec_dk.to(p_dk.dtype.element_ty), mask=mask)
        p_dq -= BK
        p_dg -= BK
        p_dk -= BK

def launch_fwd_decay_cumsum(g: Tensor, g_o: Tensor, inv_ln2: Tensor):
    BK = 32
    DK = g.shape[-1]
    BT = triton.cdiv(DK, BK)
    NT = BT
    grid = (NT, 1, g.shape[-2])
    num_warps = 1

    fwd_decay_cumsum[grid](
        g, g_o, inv_ln2,
        BT=BT, BK=BK, DK=DK,
        DK_NEXT=DK,
        BDIV_BK=triton.cdiv(DK, BK),
        TDIV_BK=triton.cdiv(DK, BK),
        BK_MIN=BK,
        num_warps=num_warps,
        num_stages=1,
    )

def launch_prepare_qg_kg(q, k, g, qg, kg):
    BK = 64
    DK = q.shape[-1]
    DG = g.shape[-1]
    BT = triton.cdiv(DK, BK)
    NT = g.shape[-2] * BT
    grid = (NT, 1, q.shape[-2])
    num_warps = 4 if DK <= 1024 else 2

    prepare_qg
