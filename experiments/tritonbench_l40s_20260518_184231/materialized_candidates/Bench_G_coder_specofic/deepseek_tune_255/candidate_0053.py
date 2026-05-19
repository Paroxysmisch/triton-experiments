import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_recurrence(
    S, d,
    O,
    NUM_HEAD,
    NUM_BLOCK,
    D_MODEL_K,
    D_MODEL_V,
    BLOCK_MODEL_K: tl.constexpr,
    BLOCK_MODEL_V: tl.constexpr,
    last_kv: tl.constexpr = None,
):
    offset_bh = tl.program_id(0)
    offset_k = tl.program_id(1) * BLOCK_MODEL_K
    offset_v = tl.program_id(2) * BLOCK_MODEL_V

    S = S + offset_bh * NUM_BLOCK * D_MODEL_K + offset_k + tl.arange(0, BLOCK_MODEL_K)
    d = d + offset_bh * NUM_BLOCK * D_MODEL_K + offset_k + tl.arange(0, BLOCK_MODEL_K)
    O = O + (offset_bh + NUM_HEAD * offset_k // D_MODEL_K * NUM_BLOCK) * D_MODEL_V + offset_v + tl.arange(0, BLOCK_MODEL_V)

    if last_kv is not None:
        last_kv = last_kv + offset_bh * NUM_BLOCK * D_MODEL_K + offset_k + tl.arange(0, BLOCK_MODEL_K)

    mask_k = (offset_k + tl.arange(0, BLOCK_MODEL_K)) < D_MODEL_K
    mask_v = (offset_v + tl.arange(0, BLOCK_MODEL_V)) < D_MODEL_V

    acc_k = tl.zeros([BLOCK_MODEL_K], dtype=tl.float32)
    acc_v = tl.zeros([BLOCK_MODEL_V], dtype=tl.float32)

    if last_kv is not None:
        _kv = tl.load(last_kv, mask=mask_k, other=0)
        acc_k += tl.sum(_kv * _kv, 0)

    for i in range(NUM_BLOCK):
        _S = tl.load(S + i * D_MODEL_K, mask=mask_k, other=0)
        _d = tl.load(d + i * D_MODEL_K, mask=mask_k, other=0)
        acc_k += tl.sum(_S * _S, 0)
        _k = acc_k * tl.math.rsqrt(acc_k + 1e-18)
        _k = tl.where(mask_k, _k, 0)
        tl.store(O + i * D_MODEL_V, _k, mask=mask_v)

        _kd = _k * _d
        acc_v += tl.sum(_kd * _kd, 0)
        _v = acc_v * tl.math.rsqrt(acc_v + 1e-18)
        _v = tl.where(mask_v, _v, 0)
        tl.store(O + i * D_MODEL_V + offset_v, _v, mask=mask_v)

@triton.jit
def _bwd_recurrence(
    S, d,
    DI, DG, DL, DS,
    NUM_HEAD,
    NUM_BLOCK,
    D_MODEL_K,
    D_MODEL_V,
    BLOCK_MODEL_K: tl.constexpr,
    BLOCK_MODEL_V: tl.constexpr,
):
    offset_bh = tl.program_id(0)
    offset_k = tl.program_id(1) * BLOCK_MODEL_K
    offset_v = tl.program_id(2) * BLOCK_MODEL_V

    S = S + offset_bh * NUM_BLOCK * D_MODEL_K + offset_k + tl.arange(0, BLOCK_MODEL_K)
    d = d + offset_bh * NUM_BLOCK * D_MODEL_K + offset_k + tl.arange(0, BLOCK_MODEL_K)
    DI = DI + offset_bh * NUM_BLOCK * D_MODEL_K + offset_k + tl.arange(0, BLOCK_MODEL_K)
    DG = DG + offset_bh * NUM_BLOCK * D_MODEL_K + offset_k + tl.arange(0, BLOCK_MODEL_K)
    DL = DL + (offset_bh + NUM_HEAD * offset_k // D_MODEL_K * NUM_BLOCK) * D_MODEL_V + offset_v + tl.arange(0, BLOCK_MODEL_V)
    DS = DS + (offset_bh + NUM_HEAD * offset_k // D_MODEL_K * NUM_BLOCK) * D_MODEL_V + offset_v + tl.arange(0, BLOCK_MODEL_V)

    mask_k = (offset_k + tl.arange(0, BLOCK_MODEL_K)) < D_MODEL_K
    mask_v = (offset_v + tl.arange(0, BLOCK_MODEL_V)) < D_MODEL_V

    acc_k = tl.zeros([BLOCK_MODEL_K], dtype=tl.float32)
    acc_v = tl.zeros([BLOCK_MODEL_V], dtype=tl.float32)

    for i in range(NUM_BLOCK - 1, -1, -1):
        _S = tl.load(S + i * D_MODEL_K, mask=mask_k, other=0)
        _d = tl.load(d + i * D_MODEL_K, mask=mask_k, other=0)
        _DI = tl.load(DI + i * D_MODEL_K, mask=mask_k, other=0)
        _DG = tl.load(DG + i * D_MODEL_K, mask=mask_k, other=0)
        _DL = tl.load(DL + i * D_MODEL_V, mask=mask_v, other=0)
        _DS = tl.load(DS + i * D_MODEL_V, mask=mask_v, other=0)

        acc_k += tl.sum(_S * _S, 0)
        _k = acc_k * tl.math.rsqrt(acc_k + 1e-18)
        _k = tl.where(mask_k, _k, 0)

        _kd = _k * _d
        acc_v += tl.sum(_kd * _kd, 0)
        _v = acc_v * tl.math.rsqrt(acc_v + 1e-18)
        _v = tl.where(mask_v, _v, 0)

        _dg = _DI * _k + _DG * _v
        tl.store(DG + i * D_MODEL_K, _dg, mask=mask_k)
        tl.store(DS + i * D_MODEL_V, _DL + _DS * _v, mask=mask_v)

        _d = _d * _k * tl.math.rsqrt(acc_k + 1e-18) - _DI * _S
        tl.store(D + i * D_MODEL_K, _d, mask=mask_k)

class ChunkGateRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cross_decay, last_kv=None):
        B, H, N, D_k, D_v = *kv.shape, kv.size(-1)
        N
