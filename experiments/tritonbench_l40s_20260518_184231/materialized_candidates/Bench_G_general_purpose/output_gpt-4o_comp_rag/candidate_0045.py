import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Literal

@triton.jit
def _div_up(val, other):
    return (val + other - 1) // other

@triton.jit
def _quant_int8(val):
    val_min = tl.min(val, 1)
    val_max = tl.max(val, 1)
    scales = (val_max - val_min) / 255
    zeros = -val_min / scales
    q_val = (val / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
    return q_val, scales, zeros

@triton.jit
def _quant_int4(val1, val2):
    val1 = val1.to(tl.float32)
    val2 = val2.to(tl.float32)
    val_min = tl.min(tl.minimum(val1, val2), 1)
    val_max = tl.max(tl.maximum(val1, val2), 1)
    scales = (val_max - val_min) / 15
    zeros = -val_min / scales
    q_val1 = (val1 / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
    q_val2 = (val2 / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
    q_val = q_val1 + q_val2 * 16
    return q_val, scales, zeros

@triton.jit
def _fill_kv_cache_kernel(
    KStates, VStates, KCaches, VCaches,
    QStartLoc, QSeqLens, KVSeqLens, BlockOffsets,
    num_heads: tl.constexpr, head_dim: tl.constexpr, head_dim_v: tl.constexpr,
    stride_kss, stride_ksh, stride_ksd, stride_vss, stride_vsh, stride_vsd,
    stride_kcn: tl.constexpr, stride_kcb: tl.constexpr, stride_kch: tl.constexpr, stride_kcd: tl.constexpr,
    stride_vcn: tl.constexpr, stride_vcb: tl.constexpr, stride_vch: tl.constexpr, stride_vcd: tl.constexpr,
    stride_boff, BLOCK: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr, BLOCK_H: tl.constexpr,
):
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    h_off = tl.arange(0, BLOCK_H)
    d_off = tl.arange(0, BLOCK_D)

    q_startloc = tl.load(QStartLoc + batch_id)
    q_seqlen = tl.load(QSeqLens + batch_id)
    kv_seqlen = tl.load(KVSeqLens + batch_id)
    history_seqlen = kv_seqlen - q_seqlen

    block0_first_tokenloc = history_seqlen % BLOCK

    state_token_offset = tl.maximum(block_id * BLOCK - block0_first_tokenloc, 0)
    kv_block_id = _div_up(history_seqlen + 1, BLOCK) - 1 + block_id
    kv_block_id = min(kv_block_id, stride_boff - 1)
    block_off = tl.load(BlockOffsets + batch_id * stride_boff + kv_block_id)

    cur_startloc = q_startloc + state_token_offset
    ks_ptr = KStates + cur_startloc * stride_kss
    vs_ptr = VStates + cur_startloc * stride_vss

    kc_ptr = KCaches + block_off * stride_kcn
    vc_ptr = VCaches + block_off * stride_vcn

    c_first_tokenloc = block0_first_tokenloc
    if block_id != 0:
        c_first_tokenloc *= 0
    c_last_tokenloc = tl.minimum(BLOCK, q_seqlen + block0_first_tokenloc - block_id * BLOCK)

    for bidx in range(c_first_tokenloc, c_last_tokenloc):
        sidx = bidx - c_first_tokenloc
        mask = (h_off[:, None] < num_heads) & (d_off[None, :] < head_dim)
        k = tl.load(ks_ptr + sidx * stride_kss + h_off[:, None] * stride_ksh + d_off[None, :] * stride_ksd, mask=mask)
        tl.store(kc_ptr + bidx * stride_kcb + h_off[:, None] * stride_kch + d_off[None, :] * stride_kcd, k, mask=mask)

        if BLOCK_DV > 0:
            dv_off = tl.arange(0, BLOCK_DV)
            maskv = (h_off[:, None] < num_heads) & (dv_off[None, :] < head_dim_v)
            v = tl.load(vs_ptr + sidx * stride_vss + h_off[:, None] * stride_vsh + dv_off[None, :] * stride_vsd, mask=maskv)
            tl.store(vc_ptr + bidx * stride_vcb + h_off[:, None] * stride_vch + dv_off[None, :] * stride_vcd, v, mask=maskv)

@triton.jit
def _fill_kv_cache_quant_kernel(
    KStates, VStates, KCaches, VCaches, KScalesZeros, VScalesZeros,
    QStartLoc, QSeqLens, KVSeqLens, BlockOffsets,
    num_heads: tl.constexpr, head_dim: tl.constexpr, head_dim_v: tl.constexpr,
    stride_kss, stride_ksh, stride_ksd, stride_vss, stride_vsh, stride_vsd,
    stride_kcn: tl.constexpr, stride_kcb: tl.constexpr, stride_kch: tl.constexpr, stride_kcd: tl.constexpr,
    stride_vcn: tl.constexpr, stride_vcb: tl.constexpr, stride_vch: tl.constexpr, stride_vcd: tl.constexpr,
    stride_kszn: tl.constexpr, stride_kszb: tl.constexpr, stride_kszh: tl.constexpr, stride_kszd: tl.constexpr,
    stride_vszn: tl.constexpr, stride_vszb: tl.constexpr, stride_vszh: tl.constexpr, stride_vszd: tl.constexpr,
    quant_policy: tl.constexpr, stride_boff, BLOCK: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr, BLOCK_H: tl.constexpr,
):
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)
    d_off = tl.arange(0, BLOCK_D)

    h_off = tl.arange(0, BLOCK_H)
    szd_off = tl.arange(0, 2)

    q_startloc = tl.load(QStartLoc + batch_id)
    q_seqlen = tl.load(QSeqLens + batch_id)
    kv_seqlen = tl.load(KVSeqLens + batch_id)
    history_seqlen = kv_seqlen - q_seqlen

    block0_first_tokenloc = history_seqlen % BLOCK

    state_token_offset = tl.maximum(block_id * BLOCK - block0_first_tokenloc, 0)
    kv_block_id = _div_up(history_seqlen + 1, BLOCK) - 1 + block_id
    kv_block_id = min(kv_block_id, stride_boff - 1)
    block_off = tl.load(BlockOffsets + batch_id * stride_boff + kv_block_id)

    cur_startloc = q_startloc + state_token_offset
    ks_ptr = KStates + cur_startloc * stride_kss
    vs_ptr = VStates + cur_startloc * stride_vss

    kc_ptr = KCaches + block_off * stride_kcn
    vc_ptr = VCaches + block_off * stride_vcn

    ksz_ptr = KScalesZeros + block_off * stride_kszn
    vsz_ptr = VScalesZeros + block_off * stride_vszn

    c_first_tokenloc = block0_first_tokenloc
    if block_id != 0:
        c_first_tokenloc *= 0
    c_last_tokenloc = tl.minimum(BLOCK, q_seqlen + block0_first_tokenloc - block_id * BLOCK)

    for bidx in range(c_first_tokenloc, c_last_tokenloc):
        sidx = bidx - c_first_tokenloc
        mask = (h_off[:, None] < num_heads) & (d_off[None, :] < head_dim)
        if quant_policy == 4:
            k1 = tl.load(ks_ptr + sidx * stride_kss + h_off[:, None] * stride_ksh + d_off[None, :] * stride_ksd, mask=mask)
            k2 = tl.load(ks_ptr + sidx * stride_kss + h_off[:, None] * stride_ksh + d_off[None, :] * stride_ksd + head_dim * stride_ksd, mask=mask)
            q_k, k_scales, k_zeros = _quant_int4(k1, k2)
        else:
            k = tl.load(ks_ptr + sidx * stride_kss + h_off[:, None] * stride_ksh + d_off[None, :] * stride_
