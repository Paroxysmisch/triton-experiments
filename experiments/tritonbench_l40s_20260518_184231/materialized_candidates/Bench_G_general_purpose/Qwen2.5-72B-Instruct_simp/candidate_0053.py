import triton
import triton.language as tl

@triton.jit
def _fwd_recurrence(S, d, O, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, last_kv, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    head_id = pid // NUM_BLOCK
    block_id = pid % NUM_BLOCK

    offset = head_id * D_MODEL_K * D_MODEL_V + block_id * BLOCK_MODEL_K * BLOCK_MODEL_V

    S_block = S + offset
    d_block = d + offset
    O_block = O + offset
    last_kv_block = last_kv + offset

    for i in range(BLOCK_SIZE):
        s = tl.load(S_block + i * BLOCK_MODEL_K)
        o = tl.load(O_block + i * BLOCK_MODEL_V)
        decay = tl.load(d_block + i * BLOCK_MODEL_K)

        if last_kv:
            last_k = tl.load(last_kv_block + i * BLOCK_MODEL_K)
            last_v = tl.load(last_kv_block + i * BLOCK_MODEL_V)
            o = decay * o + (1 - decay) * (s @ last_k)
        else:
            o = decay * o + (1 - decay) * s

        tl.store(O_block + i * BLOCK_MODEL_V, o)

@triton.jit
def _bwd_recurrence(S, d, DI, DG, DL, DS, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    head_id = pid // NUM_BLOCK
    block_id = pid % NUM_BLOCK

    offset = head_id * D_MODEL_K * D_MODEL_V + block_id * BLOCK_MODEL_K * BLOCK_MODEL_V

    S_block = S + offset
    d_block = d + offset
    DI_block = DI + offset
    DG_block = DG + offset
    DL_block = DL + offset
    DS_block = DS + offset

    for i in range(BLOCK_SIZE - 1, -1, -1):
        s = tl.load(S_block + i * BLOCK_MODEL_K)
        d = tl.load(d_block + i * BLOCK_MODEL_K)
        di = tl.load(DI_block + i * BLOCK_MODEL_V)
        dg = tl.load(DG_block + i * BLOCK_MODEL_K)
        dl = tl.load(DL_block + i * BLOCK_MODEL_V)

        ds = (1 - d) * (di @ s.T)
        dd = (1 - d) * (s @ di)
        dl = d * dl + (1 - d) * di

        tl.store(DI_block + i * BLOCK_MODEL_V, di)
        tl.store(DG_block + i * BLOCK_MODEL_K, dg + dd)
        tl.store(DL_block + i * BLOCK_MODEL_V, dl)
        tl.store(DS_block + i * BLOCK_MODEL_K, ds)

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
        grid = (self.NUM_HEAD * self.NUM_BLOCK, )
        BLOCK_SIZE = self.BLOCK_MODEL_K // self.BLOCK_MODEL_V
        _fwd_recurrence[grid](self.S, self.d, self.O, self.NUM_HEAD, self.NUM_BLOCK, self.D_MODEL_K, self.D_MODEL_V, self.BLOCK_MODEL_K, self.BLOCK_MODEL_V, self.last_kv, BLOCK_SIZE)

    def backward(self, DI, DG, DL, DS):
        grid = (self.NUM_HEAD * self.NUM_BLOCK, )
        BLOCK_SIZE = self.BLOCK_MODEL_K // self.BLOCK_MODEL_V
        _bwd_recurrence[grid](self.S, self.d, DI, DG, DL, DS, self.NUM_HEAD, self.NUM_BLOCK, self.D_MODEL_K, self.D_MODEL_V, self.BLOCK_MODEL_K, self.BLOCK_MODEL_V, BLOCK_SIZE)

import torch

# Example dimensions
NUM_HEAD = 8
NUM_BLOCK = 16
D_MODEL_K = 64
D_MODEL_V = 64
BLOCK_MODEL_K = 16
BLOCK_MODEL_V = 16

# Initialize tensors
S = torch.randn(NUM_HEAD, NUM_BLOCK, BLOCK_MODEL_K, BLOCK_MODEL_V, device='cuda')
d = torch.randn(NUM_HEAD, NUM_BLOCK, BLOCK_MODEL_K, BLOCK_MODEL_V, device='cuda')
O = torch.zeros(NUM_HEAD, NUM_BLOCK, BLOCK_MODEL_K, BLOCK_MODEL_V, device='cuda')
last_kv = torch.randn(NUM_HEAD, NUM_BLOCK, BLOCK_MODEL_K, BLOCK_MODEL_V, device='cuda')

# Initialize wrapper
chunk_gate_recurrent = ChunkGateRecurrent(S, d, O, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, last_kv)

# Forward pass
chunk_gate_recurrent.forward()

# Backward pass
DI = torch.randn_like(O)
DG = torch.randn_like(d)
DL = torch.randn_like(O)
DS = torch.zeros_like(S)
chunk_gate_recurrent.backward(DI, DG, DL, DS)
