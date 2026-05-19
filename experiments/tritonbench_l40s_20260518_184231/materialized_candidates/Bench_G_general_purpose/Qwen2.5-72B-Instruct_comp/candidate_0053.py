import triton
import triton.language as tl

@triton.jit
def _fwd_recurrence(S, d, O, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, last_kv=None, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_heads = NUM_HEAD
    num_blocks = NUM_BLOCK
    head = pid % num_heads
    block = pid // num_heads

    if last_kv is not None:
        last_k = tl.load(last_kv + head * D_MODEL_K)
        last_v = tl.load(last_kv + head * D_MODEL_K + D_MODEL_K * num_heads)

    for i in range(block * BLOCK_SIZE, (block + 1) * BLOCK_SIZE):
        if i >= num_blocks:
            break

        offset = head * D_MODEL_K + i * BLOCK_MODEL_K
        s = tl.load(S + offset)
        d_val = tl.load(d + head * num_blocks + i)

        if last_kv is not None:
            k = last_k + s * d_val
            v = last_v + s * d_val
        else:
            k = s * d_val
            v = s * d_val

        tl.store(O + offset, k)
        tl.store(O + offset + D_MODEL_K * num_heads, v)

        if last_kv is not None:
            last_k = k
            last_v = v

@triton.jit
def _bwd_recurrence(S, d, DI, DG, DL, DS, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_heads = NUM_HEAD
    num_blocks = NUM_BLOCK
    head = pid % num_heads
    block = pid // num_heads

    for i in range((block + 1) * BLOCK_SIZE - 1, block * BLOCK_SIZE - 1, -1):
        if i < 0 or i >= num_blocks:
            break

        offset = head * D_MODEL_K + i * BLOCK_MODEL_K
        s = tl.load(S + offset)
        d_val = tl.load(d + head * num_blocks + i)
        di = tl.load(DI + offset)
        dg = tl.load(DG + head * num_blocks + i)
        dl = tl.load(DL + head * D_MODEL_K + i * BLOCK_MODEL_K)

        ds = di * d_val + dg * s + dl
        tl.store(DS + offset, ds)

        if i > 0:
            next_offset = head * D_MODEL_K + (i - 1) * BLOCK_MODEL_K
            next_di = tl.load(DI + next_offset)
            next_dg = tl.load(DG + head * num_blocks + (i - 1))
            next_dl = tl.load(DL + head * D_MODEL_K + (i - 1) * BLOCK_MODEL_K)

            next_di += ds * d_val
            next_dg += ds * s
            next_dl += ds

            tl.store(DI + next_offset, next_di)
            tl.store(DG + head * num_blocks + (i - 1), next_dg)
            tl.store(DL + head * D_MODEL_K + (i - 1) * BLOCK_MODEL_K, next_dl)

import torch
from torch.autograd import Function

class ChunkGateRecurrent(Function):
    @staticmethod
    def forward(ctx, kv, cross_decay, last_kv=None):
        NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V = kv.shape[0], kv.shape[1], kv.shape[2], kv.shape[3]
        BLOCK_MODEL_K, BLOCK_MODEL_V = D_MODEL_K // NUM_BLOCK, D_MODEL_V // NUM_BLOCK

        O = torch.empty_like(kv)
        grid = (NUM_HEAD * NUM_BLOCK,)

        _fwd_recurrence[grid](
            kv, cross_decay, O, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, last_kv, BLOCK_SIZE=1
        )

        ctx.save_for_backward(kv, cross_decay, O, last_kv)
        return O

    @staticmethod
    def backward(ctx, DO):
        kv, cross_decay, O, last_kv = ctx.saved_tensors
        NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V = kv.shape[0], kv.shape[1], kv.shape[2], kv.shape[3]
        BLOCK_MODEL_K, BLOCK_MODEL_V = D_MODEL_K // NUM_BLOCK, D_MODEL_V // NUM_BLOCK

        DI = torch.zeros_like(kv)
        DG = torch.zeros((NUM_HEAD, NUM_BLOCK))
        DL = torch.zeros_like(kv)
        DS = torch.empty_like(kv)

        grid = (NUM_HEAD * NUM_BLOCK,)

        _bwd_recurrence[grid](
            kv, cross_decay, DI, DG, DL, DS, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, BLOCK_SIZE=1
        )

        return DS, DG, DL, None

# Example usage
NUM_HEAD = 4
NUM_BLOCK = 8
D_MODEL_K = 64
D_MODEL_V = 64
BLOCK_MODEL_K = D_MODEL_K // NUM_BLOCK
BLOCK_MODEL_V = D_MODEL_V // NUM_BLOCK

kv = torch.randn(NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, device='cuda')
cross_decay = torch.randn(NUM_HEAD, NUM_BLOCK, device='cuda')
last_kv = torch.randn(NUM_HEAD, D_MODEL_K, D_MODEL_V, device='cuda')

output = ChunkGateRecurrent.apply(kv, cross_decay, last_kv)

# Backward pass
output.backward(torch.randn_like(output))
