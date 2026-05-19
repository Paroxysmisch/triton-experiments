import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_recurrence(
    S, d,
    O,
    NUM_HEAD, NUM_BLOCK,
    D_MODEL_K, D_MODEL_V,
    BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr,
    last_kv
):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_k = tl.program_id(2)

    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + \
        tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_k * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[None, :]

    d = d + offset_bh * NUM_BLOCK * D_MODEL_K + offset_d * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)

    O = O + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + \
        tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_k * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[None, :] + \
        offset_bh * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]

    if last_kv is not None:
        last_kv = last_kv + offset_bh * D_MODEL_K * D_MODEL_V + offset_k * D_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]

    acc = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)
    if last_kv is not None:
        acc += tl.dot(tl.load(S), tl.load(last_kv))
    S_roof = tl.load(S + D_MODEL_K * D_MODEL_V)
    acc += tl.dot(tl.load(S + D_MODEL_K * D_MODEL_V).to(tl.float32), tl.load(last_kv).to(tl.float32))
    for i in range(tl.num_programs(3) - 2):
        S_roof = tl.load(S + i * D_MODEL_K * D_MODEL_V)
        acc = acc * tl.load(d + i * BLOCK_MODEL_K)[:, None] + tl.dot(S_roof.to(tl.float32), tl.load(last_kv).to(tl.float32))
        tl.store(O, acc.to(O.dtype.element_ty))
        O += D_MODEL_K * D_MODEL_V
    for i in range(tl.num_programs(3) - 2):
        S = S + D_MODEL_K * D_MODEL_V
        last_kv = last_kv + D_MODEL_K * D_MODEL_V
        O = O + D_MODEL_K * D_MODEL_V
        S_roof = tl.load(S)
        acc = acc * tl.load(d + i * BLOCK_MODEL_K)[:, None] + tl.dot(S_roof.to(tl.float32), tl.load(last_kv).to(tl.float32))
        tl.store(O, acc.to(O.dtype.element_ty))

@triton.jit
def _bwd_recurrence(
    S, d,
    DI, DG, DL, DS,
    NUM_HEAD, NUM_BLOCK,
    D_MODEL_K, D_MODEL_V,
    BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr,
):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_k = tl.program_id(2)

    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + \
        tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_k * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[None, :] + \
        (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    d = d + offset_bh * NUM_BLOCK * D_MODEL_K + offset_d * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K) + (NUM_BLOCK - 2) * D_MODEL_K

    DS = DS + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + \
        tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_k * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[None, :] + \
        (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    DI = DI + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL_K + \
        tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_k * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[None, :] + \
        offset_bh * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]

    DG = DG + offset_bh * NUM_BLOCK * D_MODEL_K + offset_d * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K) + \
        offset_bh * D_MODEL_K + offset_d * D_MODEL_V + tl.arange(0, BLOCK_MODEL_V)

    DL = DL + offset_bh * NUM_BLOCK * D_MODEL_K + offset_d * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K) + \
        offset_bh * D_MODEL_K + offset_d * D_MODEL_V + tl.arange(0, BLOCK_MODEL_V)

    S_roof = tl.load(S + D_MODEL_K * D_MODEL_V).to(tl.float32)
    DS_roof = tl.load(DS + D_MODEL_K * D_MODEL_V).to(tl.float32)
    d_roof = tl.load(d + BLOCK_MODEL_K).to(tl.float32)
    acc_g = tl.sum(S_roof * DS_roof, axis=1)
    acc_l = tl.sum(S_roof * DS_roof, axis=0)
    for i in range(tl.num_programs(3) - 1):
        S = S - D_MODEL_K * D_MODEL_V
        DS = DS - D_MODEL_K * D_MODEL_V
        d = d - BLOCK_MODEL_K
        DG = DG - BLOCK_MODEL_K
        DL = DL - BLOCK_MODEL_K
        S_roof = tl.load(S).to(tl.float32)
        DS_roof = tl.load(DS).to(tl.float32)
        d_roof = tl.load(d).to(tl.float32)
        acc_g = acc_g * d_roof + tl.sum(S_roof * DS_roof, axis=1)
        acc_l = acc_l * d_roof + tl.sum(S_roof * DS_roof, axis=0)
        tl.store(DG, acc_g.to(DG.dtype.element_ty))
        tl.store(DL, acc_l.to(DL.dtype.element_ty))
    tl.store(DG, acc_g.to(DG.dtype.element_ty))
    tl.store(DL, acc_l.to(DL.dtype.element_ty))

    S = S + (tl.num_programs(3) - 1) * D_MODEL_K * D_MODEL_V
    d = d + (tl.num_programs(3) - 2) * BLOCK_MODEL_K
    DS = DS + (tl.num_programs(3) - 1) * D_MODEL_K * D_MODEL_V
    last_d = tl.load(d + BLOCK_MODEL_K).to(tl.float32)

    acc = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)
    for i in range(tl.num_programs(3) - 1):
        S_roof = tl.load(S)
        DS_roof = tl.load(DS)
        acc = acc * last_d[:, None] + tl.dot(S_roof.to(tl.float32), DS_roof.to(tl.float32))
        tl.store(DI, acc.to(DI.dtype.element_ty))
        DI += D_MODEL_K * D_MODEL_V
        S -= D_MODEL_K * D_MODEL_V
        DS -= D_MODEL_K * D_MODEL_V
    S_roof = tl.load(S)
    DS_roof = tl.load(DS)
    acc = acc * last_d[:, None] + tl.dot(S_roof.to(tl.float32), DS_roof.to(tl.float32))
    tl.store(DI, acc.to(DI.dtype.element_ty))

class ChunkGateRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cross_decay, last_kv=None):
        BLOCK_MODEL_K, BLOCK_MODEL_V = 16, 64
        NUM_HEAD, NUM_BLOCK = kv.shape[1], kv.shape[2]
        D_MODEL_K, D_MODEL_V = NUM_HEAD * BLOCK_MODEL_K, NUM_HEAD * BLOCK_MODEL_V
        seq_len = kv.shape[0]
        ori_kv_shape = kv.shape
        kv = kv.view(-1, NUM_HEAD, NUM_BLOCK, BLOCK_MODEL_K, BLOCK_MODEL_V)
        kv = kv.permute(1, 2, 0, 3, 4).contiguous()
        o = torch.empty_like(kv[:, :, 1:, :, :]).to(kv.dtype)
        if last_kv is not None:
            last_kv = last_kv.view(-1, NUM_HEAD, BLOCK_MODEL_K, BLOCK_MODEL_V)
            last_kv = last_kv.permute(1, 0, 2, 3).contiguous()
