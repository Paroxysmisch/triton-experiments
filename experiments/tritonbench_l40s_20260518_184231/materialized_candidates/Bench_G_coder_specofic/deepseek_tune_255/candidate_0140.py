import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def fwd_decay_cumsum(
    g: tl.tensor,
    g_o: tl.tensor,
    decay: tl.tensor,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    inv_ln2: tl.constexpr,
):
    i_k, i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3)
    i_d = tl.arange(0, BT)
    cum_decay = tl.zeros([BK, BV], dtype=tl.float32)
    p_g = tl.make_block_ptr(g + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK + i_d, i_v * BV + i_d), (BT, BT), (1, 0))
    p_g_o = tl.make_block_ptr(g_o + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK + i_d, i_v * BV + i_d), (BT, BT), (1, 0))
    mask = (i_k * BK + i_d < DK) & (i_v * BV + i_d < DV)
    # [BT, BT]
    g_block = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    g_block *= inv_ln2
    cum_decay += tl.sum(g_block, axis=1)[:, None]
    tl.store(p_g_o, cum_decay.to(p_g_o.dtype.element_ty), boundary_check=(0, 1))

def fwd_decay_cumsum_launch(g: Tensor, g_o: Tensor, BT: int, BK: int, BV: int, inv_ln2: float):
    DK, DV = g.shape[-2:]
    T = DV // BV
    B, H = g.shape[:2]
    grid = (B * H, DV // BV, T // BT, 1)
    fwd_decay_cumsum[grid](g, g_o, BT, BK, BV, DK, DV, T, B, H, inv_ln2)

@triton.jit
def prepare_qg_kg(
    q: tl.tensor,
    k: tl.tensor,
    g: tl.tensor,
    qg: tl.tensor,
    kg: tl.tensor,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
):
    i_k, i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3)
    i_d = tl.arange(0, BT)
    p_q = tl.make_block_ptr(q + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK + i_d, i_v * BV + i_d), (BT, BT), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK + i_d, i_v * BV + i_d), (BT, BT), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK + i_d, i_v * BV + i_d), (BT, BT), (1, 0))
    p_qg = tl.make_block_ptr(qg + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK + i_d, i_v * BV + i_d), (BT, BT), (1, 0))
    p_kg = tl.make_block_ptr(kg + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK + i_d, i_v * BV + i_d), (BT, BT), (1, 0))
    mask = (i_k * BK + i_d < DK) & (i_v * BV + i_d < DV)
    # [BT, BT]
    q_block = tl.load(p_q, boundary_check=(0, 1))
    k_block = tl.load(p_k, boundary_check=(0, 1))
    g_block = tl.load(p_g, boundary_check=(0, 1))
    g_block = tl.exp(g_block)
    tl.store(p_kg, g_block.to(p_kg.dtype.element_ty), boundary_check=(0, 1))
    qg_block = q_block * g_block
    tl.store(p_qg, qg_block.to(p_qg.dtype.element_ty), boundary_check=(0, 1))

def prepare_qg_kg_launch(q: Tensor, k: Tensor, g: Tensor, qg: Tensor, kg: Tensor, BT: int, BK: int, BV: int):
    DK, DV = q.shape[-2:]
    T = DV // BV
    B, H = q.shape[:2]
    grid = (B * H, DV // BV, T // BT, 1)
    prepare_qg_kg[grid](q, k, g, qg, kg, BT, BK, BV, DK, DV, T, B, H)

@triton.jit
def bwd_decay_global_cumsum(
    dq_inner: tl.tensor,
    dq_inter: tl.tensor,
    dk_inner: tl.tensor,
    dk_inter: tl.tensor,
    q: tl.tensor,
    k: tl.tensor,
    g: tl.tensor,
    dg: tl.tensor,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
):
    i_k, i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3)
    i_d = tl.arange(0, BT)
    p_q = tl.make_block_ptr(q + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK + i_d, i_v * BV + i_d), (BT, BT), (1, 0))
