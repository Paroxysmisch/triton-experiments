import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_recurrence(
    S, d, O,  # Main tensors
    NUM_HEAD: tl.constexpr, NUM_BLOCK: tl.constexpr,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr,
    last_kv=None
):
    # Get program ID
    pid = tl.program_id(0)
    head_id = pid // NUM_BLOCK
    block_id = pid % NUM_BLOCK

    # Initialize accumulators
    acc_k = tl.zeros([BLOCK_MODEL_K], dtype=tl.float32)
    acc_v = tl.zeros([BLOCK_MODEL_V], dtype=tl.float32)
    
    # Load last key-value if provided
    if last_kv is not None:
        last_k = tl.load(last_kv + head_id * D_MODEL_K + 
                        block_id * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K))
        last_v = tl.load(last_kv + NUM_HEAD * D_MODEL_K + head_id * D_MODEL_V + 
                        block_id * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V))
        acc_k = last_k
        acc_v = last_v

    # Main recurrence loop
    for i in range(0, NUM_BLOCK):
        # Load decay factor
        decay = tl.load(d + head_id * NUM_BLOCK + i)
        
        # Load current block
        offset_k = (head_id * NUM_BLOCK + i) * BLOCK_MODEL_K
        offset_v = (head_id * NUM_BLOCK + i) * BLOCK_MODEL_V
        
        curr_k = tl.load(S + offset_k + tl.arange(0, BLOCK_MODEL_K))
        curr_v = tl.load(S + NUM_HEAD * D_MODEL_K + offset_v + 
                        tl.arange(0, BLOCK_MODEL_V))

        # Apply decay and accumulate
        acc_k = acc_k * tl.exp(decay) + curr_k
        acc_v = acc_v * tl.exp(decay) + curr_v

        # Store output
        offset_o = (head_id * NUM_BLOCK + i) * BLOCK_MODEL_V
        tl.store(O + offset_o + tl.arange(0, BLOCK_MODEL_V), acc_v)

@triton.jit
def _bwd_recurrence(
    S, d, DI, DG, DL, DS,  # Tensors
    NUM_HEAD: tl.constexpr, NUM_BLOCK: tl.constexpr,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    head_id = pid // NUM_BLOCK
    block_id = pid % NUM_BLOCK

    # Initialize gradient accumulators
    grad_k = tl.zeros([BLOCK_MODEL_K], dtype=tl.float32)
    grad_v = tl.zeros([BLOCK_MODEL_V], dtype=tl.float32)
    grad_d = tl.zeros([1], dtype=tl.float32)

    # Backward pass loop
    for i in range(NUM_BLOCK-1, -1, -1):
        # Load gradients
        offset = (head_id * NUM_BLOCK + i) * BLOCK_MODEL_V
        curr_grad = tl.load(DG + offset + tl.arange(0, BLOCK_MODEL_V))
        
        # Load decay factor
        decay = tl.load(d + head_id * NUM_BLOCK + i)

        # Compute gradients
        grad_k = grad_k * tl.exp(decay)
        grad_v = grad_v * tl.exp(decay)
        
        # Accumulate input gradients
        offset_k = (head_id * NUM_BLOCK + i) * BLOCK_MODEL_K
        offset_v = (head_id * NUM_BLOCK + i) * BLOCK_MODEL_V
        
        tl.store(DI + offset_k + tl.arange(0, BLOCK_MODEL_K), grad_k)
        tl.store(DI + NUM_HEAD * D_MODEL_K + offset_v + 
                tl.arange(0, BLOCK_MODEL_V), grad_v)

        # Accumulate decay gradients
        grad_d[0] += tl.sum(grad_k * curr_grad)
        tl.store(DL + head_id * NUM_BLOCK + i, grad_d)

class ChunkGateRecurrent(torch.nn.Module):
    def __init__(self, num_heads, num_blocks, d_model_k, d_model_v,
                 block_model_k, block_model_v):
        super().__init__()
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.d_model_k = d_model_k
        self.d_model_v = d_model_v
        self.block_model_k = block_model_k
        self.block_model_v = block_model_v

    def forward(self, S, d, last_kv=None):
        batch_size = S.size(0)
        O = torch.zeros((batch_size, self.num_heads, self.num_blocks, 
                        self.block_model_v), device=S.device)
        
        grid = (self.num_heads * self.num_blocks,)
        _fwd_recurrence[grid](
            S, d, O,
            self.num_heads, self.num_blocks,
            self.d_model_k, self.d_model_v,
            self.block_model_k, self.block_model_v,
            last_kv
        )
        return O

    def backward(self, grad_output, S, d):
        batch_size = S.size(0)
        DI = torch.zeros_like(S)
        DG = grad_output
        DL = torch.zeros((batch_size, self.num_heads, self.num_blocks), 
                        device=S.device)
        DS = torch.zeros_like(S)
        
        grid = (self.num_heads * self.num_blocks,)
        _bwd_recurrence[grid](
            S, d, DI, DG, DL, DS,
            self.num_heads, self.num_blocks,
            self.d_model_k, self.d_model_v,
            self.block_model_k, self.block_model_v
        )
        return DI, DL, DS
