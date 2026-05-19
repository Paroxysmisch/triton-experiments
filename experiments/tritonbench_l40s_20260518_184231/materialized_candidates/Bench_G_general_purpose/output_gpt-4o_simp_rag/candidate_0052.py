import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_recurrence(S, d, O, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, last_kv):
    # Define block indices
    i_h, i_b = tl.program_id(0), tl.program_id(1)
    
    # Initialize output and temporary variables
    o = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)
    if last_kv:
        kv = tl.load(last_kv + i_h * D_MODEL_K * D_MODEL_V, mask=i_b < NUM_BLOCK)
    else:
        kv = tl.zeros([D_MODEL_K, D_MODEL_V], dtype=tl.float32)

    for b in range(NUM_BLOCK):
        s = tl.load(S + i_h * D_MODEL_K * NUM_BLOCK + b * BLOCK_MODEL_K, mask=i_b < NUM_BLOCK)
        decay = tl.load(d + i_h * NUM_BLOCK + b, mask=i_b < NUM_BLOCK)
        
        kv = kv * tl.exp2(decay) + s
        o = tl.dot(kv, tl.trans(s))
        
        tl.store(O + i_h * D_MODEL_V * NUM_BLOCK + b * BLOCK_MODEL_V, o, mask=i_b < NUM_BLOCK)

@triton.jit
def _bwd_recurrence(S, d, DI, DG, DL, DS, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V):
    # Define block indices
    i_h, i_b = tl.program_id(0), tl.program_id(1)
    
    # Initialize gradients and temporary variables
    grad_kv = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)
    
    for b in range(NUM_BLOCK - 1, -1, -1):
        s = tl.load(S + i_h * D_MODEL_K * NUM_BLOCK + b * BLOCK_MODEL_K, mask=i_b < NUM_BLOCK)
        decay = tl.load(d + i_h * NUM_BLOCK + b, mask=i_b < NUM_BLOCK)
        
        grad_s = tl.dot(tl.trans(grad_kv), s)
        grad_decay = tl.sum(grad_kv * s)
        
        tl.store(DS + i_h * D_MODEL_K * NUM_BLOCK + b * BLOCK_MODEL_K, grad_s, mask=i_b < NUM_BLOCK)
        tl.store(DG + i_h * NUM_BLOCK + b, grad_decay, mask=i_b < NUM_BLOCK)
        
        grad_kv = grad_kv * tl.exp2(decay) + s

class ChunkGateRecurrent:
    def __init__(self, num_head, num_block, d_model_k, d_model_v, block_model_k, block_model_v):
        self.num_head = num_head
        self.num_block = num_block
        self.d_model_k = d_model_k
        self.d_model_v = d_model_v
        self.block_model_k = block_model_k
        self.block_model_v = block_model_v

    def forward(self, S, d, last_kv=None):
        O = torch.empty_like(S)
        grid = (self.num_head, self.num_block)
        _fwd_recurrence[grid](S, d, O, self.num_head, self.num_block, self.d_model_k, self.d_model_v, self.block_model_k, self.block_model_v, last_kv)
        return O

    def backward(self, S, d, DI, DG, DL, DS):
        grid = (self.num_head, self.num_block)
        _bwd_recurrence[grid](S, d, DI, DG, DL, DS, self.num_head, self.num_block, self.d_model_k, self.d_model_v, self.block_model_k, self.block_model_v)
