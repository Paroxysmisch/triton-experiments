import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_recurrence(
    S, d, O,
    NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V,
    BLOCK_MODEL_K, BLOCK_MODEL_V,
    last_kv: tl.constexpr
):
    head_id = tl.program_id(0)
    block_id = tl.program_id(1)

    if head_id >= NUM_HEAD or block_id >= NUM_BLOCK:
        return

    o_ptr = tl.make_block_ptr(O, (NUM_BLOCK, D_MODEL_V), (1, BLOCK_MODEL_V), (block_id, 0), (1, BLOCK_MODEL_V), (0, 1))
    s_ptr = tl.make_block_ptr(S, (NUM_BLOCK, D_MODEL_K), (1, BLOCK_MODEL_K), (block_id, 0), (1, BLOCK_MODEL_K), (0, 1))
    d_ptr = tl.make_block_ptr(d, (NUM_BLOCK, D_MODEL_K), (1, BLOCK_MODEL_K), (block_id, 0), (1, BLOCK_MODEL_K), (0, 1))

    if last_kv:
        last_k_ptr = tl.make_block_ptr(last_kv, (1, D_MODEL_K), (1, BLOCK_MODEL_K), (0, 0), (1, BLOCK_MODEL_K), (0, 1))
        last_v_ptr = tl.make_block_ptr(last_kv, (1, D_MODEL_V), (1, BLOCK_MODEL_V), (0, 0), (1, BLOCK_MODEL_V), (0, 1))
        last_k = tl.load(last_k_ptr, boundary_check=(0, 1))
        last_v = tl.load(last_v_ptr, boundary_check=(0, 1))

    for i in range(0, NUM_BLOCK):
        s = tl.load(s_ptr, boundary_check=(0, 1))
        d_val = tl.load(d_ptr, boundary_check=(0, 1))

        if last_kv and i == 0:
            o_val = tl.dot(s, last_v, allow_tf32=False)
            s = s * tl.math.exp2(d_val) + last_k
        else:
            o_val = tl.dot(s, s, allow_tf32=False)
            s = s * tl.math.exp2(d_val) + s

        tl.store(o_ptr, o_val.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))

        s_ptr = tl.advance(s_ptr, (1, 0))
        d_ptr = tl.advance(d_ptr, (1, 0))
        o_ptr = tl.advance(o_ptr, (1, 0))

@triton.jit
def _bwd_recurrence(
    S, d, DI, DG, DL, DS,
    NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V,
    BLOCK_MODEL_K, BLOCK_MODEL_V
):
    head_id = tl.program_id(0)
    block_id = tl.program_id(1)

    if head_id >= NUM_HEAD or block_id >= NUM_BLOCK:
        return

    di_ptr = tl.make_block_ptr(DI, (NUM_BLOCK, D_MODEL_V), (1, BLOCK_MODEL_V), (block_id, 0), (1, BLOCK_MODEL_V), (0, 1))
    dg_ptr = tl.make_block_ptr(DG, (NUM_BLOCK, D_MODEL_K), (1, BLOCK_MODEL_K), (block_id, 0), (1, BLOCK_MODEL_K), (0, 1))
    dl_ptr = tl.make_block_ptr(DL, (NUM_BLOCK, D_MODEL_K), (1, BLOCK_MODEL_K), (block_id, 0), (1, BLOCK_MODEL_K), (0, 1))
    ds_ptr = tl.make_block_ptr(DS, (NUM_BLOCK, D_MODEL_K), (1, BLOCK_MODEL_K), (block_id, 0), (1, BLOCK_MODEL_K), (0, 1))
    s_ptr = tl.make_block_ptr(S, (NUM_BLOCK, D_MODEL_K), (1, BLOCK_MODEL_K), (block_id, 0), (1, BLOCK_MODEL_K), (0, 1))
    d_ptr = tl.make_block_ptr(d, (NUM_BLOCK, D_MODEL_K), (1, BLOCK_MODEL_K), (block_id, 0), (1, BLOCK_MODEL_K), (0, 1))

    for i in range(NUM_BLOCK - 1, -1, -1):
        di = tl.load(di_ptr, boundary_check=(0, 1))
        s = tl.load(s_ptr, boundary_check=(0, 1))
        d_val = tl.load(d_ptr, boundary_check=(0, 1))

        if i == NUM_BLOCK - 1:
            ds = tl.dot(s, di, allow_tf32=False)
            dg = s * tl.math.exp2(d_val) * ds
            dl = tl.dot(s, ds, allow_tf32=False)
        else:
            ds = tl.dot(s, di, allow_tf32=False)
            dg = s * tl.math.exp2(d_val) * ds
            dl = tl.dot(s, ds, allow_tf32=False)

        tl.store(ds_ptr, ds.to(ds_ptr.dtype.element_ty), boundary_check=(0, 1))
        tl.store(dg_ptr, dg.to(dg_ptr.dtype.element_ty), boundary_check=(0, 1))
        tl.store(dl_ptr, dl.to(dl_ptr.dtype.element_ty), boundary_check=(0, 1))

        di_ptr = tl.advance(di_ptr, (-1, 0))
        dg_ptr = tl.advance(dg_ptr, (-1, 0))
        dl_ptr = tl.advance(dl_ptr, (-1, 0))
        ds_ptr = tl.advance(ds_ptr, (-1, 0))
        s_ptr = tl.advance(s_ptr, (-1, 0))
        d_ptr = tl.advance(d_ptr, (-1, 0))

import torch

class ChunkGateRecurrent:
    def __init__(self, S, d, O, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, last_kv=None):
        self.S = S
        self.d = d
        self.O = O
        self.NUM_HEAD = NUM_HEAD
        self.NUM_BLOCK = NUM_BLOCK
        self.D_MODEL_K = D_MODEL_K
        self.D_MODEL_V = D_MODEL_V
        self.BLOCK_MODEL_K = BLOCK_MODEL_K
        self.BLOCK_MODEL_V = BLOCK_MODEL_V
        self.last_kv = last_kv

    def forward(self):
        grid = (self.NUM_HEAD, self.NUM_BLOCK)
        _fwd_recurrence[grid](
            self.S, self.d, self.O,
            self.NUM_HEAD, self.NUM_BLOCK, self.D_MODEL_K, self.D_MODEL_V,
            self.BLOCK_MODEL_K, self.BLOCK_MODEL_V,
            last_kv=self.last_kv is not None
        )
        return self.O

    def backward(self, DI, DG, DL, DS):
        grid = (self.NUM_HEAD, self.NUM_BLOCK)
        _bwd_recurrence[grid](
            self.S, self.d, DI, DG, DL, DS,
            self.NUM_HEAD, self.NUM_BLOCK, self.D_MODEL_K, self.D_MODEL_V,
            self.BLOCK_MODEL_K, self.BLOCK_MODEL_V
        )
        return DI, DG, DL, DS

# Example usage
B, H, T, D_MODEL_K, D_MODEL_V = 2, 4, 16, 64, 64
BLOCK_MODEL_K, BLOCK_MODEL_V = 32, 32
NUM_BLOCK = T // BLOCK_MODEL_K

S = torch.randn(B, H, T, D_MODEL_K, device='cuda')
d = torch.randn(B, H, T, D_MODEL_K, device='cuda')
O = torch.zeros(B, H, T, D_MODEL_V, device='cuda')
last_kv = torch.randn(B, H, D_MODEL_K + D_MODEL_V, device='cuda')

recurrent_op = ChunkGateRecurrent(S, d, O, H, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, last_kv=last_kv)

# Forward pass
O = recurrent_op.forward()

# Backward pass
DI = torch.randn_like(O)
DG = torch.randn_like(d)
DL = torch.randn_like(S)
DS = torch.zeros_like(S)
DI, DG, DL, DS = recurrent_op.backward(DI, DG, DL, DS)
