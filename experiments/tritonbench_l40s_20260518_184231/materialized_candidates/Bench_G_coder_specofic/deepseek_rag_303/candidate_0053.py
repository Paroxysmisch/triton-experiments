import torch
import triton
import triton.language as tl
import numpy as np

@triton.jit
def _fwd_recurrence(
    S, d,
    O,
    NUM_HEAD,
    NUM_BLOCK,
    D_MODEL_K,
    D_MODEL_V,
    BLOCK_MODEL_K,
    BLOCK_MODEL_V,
    last_kv: tl.constexpr,
):
    offset_bh = tl.program_id(0)
    offset_s = tl.program_id(1)
    offset_block_k = tl.program_id(2)
    offset_k = tl.arange(0, BLOCK_MODEL_K)
    offset_v = tl.arange(0, BLOCK_MODEL_V)

    p_d = d + offset_bh * D_MODEL_K * D_MODEL_V + (offset_s * ((NUM_BLOCK - 1) * D_MODEL_K) +
                                                   (((offset_block_k + offset_block_k * offset_k) * D_MODEL_V) + offset_v))
    p_o = O + offset_bh * D_MODEL_K * D_MODEL_V + (
        offset_s * ((NUM_BLOCK - 1) * D_MODEL_K) +
        ((((offset_block_k + offset_block_k * offset_k) * D_MODEL_V) + offset_v + int(D_MODEL_V / 2)) % (D_MODEL_K * D_MODEL_V)
         ))
    p_s = S + offset_bh * D_MODEL_K * D_MODEL_V + (
        offset_s * ((NUM_BLOCK - 1) * D_MODEL_K) +
        (((offset_block_k + offset_block_k * offset_k) * D_MODEL_V) + offset_v + int(D_MODEL_V / 2)) % (D_MODEL_K * D_MODEL_V)
         )
    mask = offset_v < ((offset_block_k + 1) * BLOCK_MODEL_K) % D_MODEL_V

    if last_kv:
        p_prev_kv = last_kv + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + \
                    ((offset_block_k - 1) * BLOCK_MODEL_K * D_MODEL_V + offset_k[:, None] * D_MODEL_V +
                     offset_v[None, :])
        prev_kv = tl.load(p_prev_kv, mask=mask[None, :], other=0).to(tl.float32)
    else:
        prev_kv = tl.zeros([D_MODEL_K, D_MODEL_V], dtype=tl.float32)

    block_k = offset_block_k + 1
    for i in range(offset_s, NUM_BLOCK):
        p_s_i = tl.load(p_s + (i * D_MODEL_K + offset_k[:, None]) * D_MODEL_V +
                        offset_v[None, :])
        p_d_i = tl.load(p_d + (i * D_MODEL_K + offset_k[:, None]) * D_MODEL_V +
                        offset_v[None, :])

        block_accumulator = tl.zeros([D_MODEL_K, D_MODEL_V], dtype=tl.float32)
        for j in range(0, block_k).reversed:
            _max = tl.max(prev_kv * p_d_i, 1)
            _exp = tl.exp2(_max - _max.max(0))
            block_output = (p_d_i * _exp[:, None]).to(p_s_i.dtype)
            block_accumulator = block_accumulator * tl.exp2(_max.max(0) - _max) + block_output

        tl.store(p_o, block_accumulator.to(p_o.dtype.element_ty), mask=mask[None, :])

        p_o += D_MODEL_K * D_MODEL_V
        p_s += (D_MODEL_K + BLOCK_MODEL_K) * D_MODEL_V
        prev_kv = block_accumulator

@triton.jit
def _bwd_recurrence(
    S, d,
    DI, DG, DL, DS,
    NUM_HEAD,
    NUM_BLOCK,
    D_MODEL_K,
    D_MODEL_V,
    BLOCK_MODEL_K,
    BLOCK_MODEL_V):
    # reverse offsetting
    offset_bh = tl.program_id(0)
    offset_s = tl.program_id(1) + NUM_BLOCK - 1
    offset_block_k = tl.program_id(2)
    offset_k = tl.arange(0, BLOCK_MODEL_K)
    offset_v = tl.arange(0, BLOCK_MODEL_V)

    # layout
    p_s = S + offset_bh * D_MODEL_K * D_MODEL_V + (
        offset_s * ((NUM_BLOCK - 1) * D_MODEL_K) +
        (((offset_block_k + offset_block_k * offset_k) * D_MODEL_V) + offset_v + int(D_MODEL_V / 2)) %
        (D_MODEL_K * D_MODEL_V))
    p_d = d + offset_bh * D_MODEL_K * D_MODEL_V + (
        offset_s * ((NUM_BLOCK - 1) * D_MODEL_K) +
        (((offset_block_k + offset_block_k * offset_k) * D_MODEL_V) + offset_v)
        )
    p_di = DI + offset_bh * D_MODEL_K * D_MODEL_V + (
        offset_s * ((NUM_BLOCK - 1) * D_MODEL_K) +
        (((offset_block_k + offset_block_k * offset_k) * D_MODEL_V) + offset_v + int(D_MODEL_V / 2)) %
        (D_MODEL_K * D_MODEL_V))
    p_dg = DG + offset_bh * D_MODEL_K * D_MODEL_V + (
        offset_s * ((NUM_BLOCK - 1) * D_MODEL_K) +
        (((offset_block_k + offset_block_k * offset_k) * D_MODEL_V) + offset_v + int(D_MODEL_V / 2)) %
        (D_MODEL_K * D_MODEL_V))
    p_ds = DS + offset_bh * D_MODEL_K * D_MODEL_V + (
        offset_s * ((NUM_BLOCK - 1) * D_MODEL_K) +
        (((offset_block_k + offset_block_k * offset_k) * D_MODEL_V) + offset_v))
    mask = offset_v < ((offset_block_k + 1) * BLOCK_MODEL_K) % D_MODEL_V

    s_i = tl.load(p_s + (offset_k[:, None] * D_MODEL_V + offset_v[None, :]))
    d_i = tl.load(p_d + (offset_k[:, None] * D_MODEL_V + offset_v[None, :]))
    block_accumulator = tl.zeros([D_MODEL_K, D_MODEL_V], dtype=tl.float32)
    for i in range(offset_s, -1, -1):
        p_s_prev = S + offset_bh *
