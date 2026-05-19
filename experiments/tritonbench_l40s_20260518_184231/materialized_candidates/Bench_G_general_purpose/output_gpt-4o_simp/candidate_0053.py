import triton
import triton.language as tl

@triton.jit
def _fwd_recurrence(S, d, O, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, last_kv, stride_s, stride_d, stride_o, stride_last_kv, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    head_idx = pid // NUM_BLOCK
    block_idx = pid % NUM_BLOCK

    # Load the decay coefficient for this block
    decay = tl.load(d + block_idx)

    # Iterate over blocks of keys/values
    for i in range(0, BLOCK_SIZE, BLOCK_MODEL_K):
        # Compute the indices for the block
        block_start = block_idx * BLOCK_MODEL_K + i

        # Load the current key/value chunk
        kv_chunk = tl.load(S + head_idx * stride_s + block_start, mask=block_start < D_MODEL_K)

        # Optionally use the last key/value
        if last_kv is not None:
            kv_chunk = kv_chunk + tl.load(last_kv + head_idx * stride_last_kv + block_start, mask=block_start < D_MODEL_K)

        # Apply the decay transformation
        kv_chunk = kv_chunk * decay

        # Store the result in the output tensor
        tl.store(O + head_idx * stride_o + block_start, kv_chunk, mask=block_start < D_MODEL_K)

@triton.jit
def _bwd_recurrence(S, d, DI, DG, DL, DS, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, stride_s, stride_d, stride_di, stride_dg, stride_dl, stride_ds, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    head_idx = pid // NUM_BLOCK
    block_idx = pid % NUM_BLOCK

    # Load the decay coefficient for this block
    decay = tl.load(d + block_idx)

    # Iterate over blocks in reverse order for backpropagation
    for i in range(BLOCK_SIZE - BLOCK_MODEL_K, -1, -BLOCK_MODEL_K):
        block_start = block_idx * BLOCK_MODEL_K + i

        # Load the current key/value chunk
        kv_chunk = tl.load(S + head_idx * stride_s + block_start, mask=block_start < D_MODEL_K)

        # Compute gradients
        grad_kv_chunk = tl.load(DI + head_idx * stride_di + block_start, mask=block_start < D_MODEL_K)
        grad_kv_chunk = grad_kv_chunk * decay

        # Accumulate gradients
        tl.atomic_add(DG + block_idx, grad_kv_chunk)

        # Store the result in the gradient tensor
        tl.store(DS + head_idx * stride_ds + block_start, grad_kv_chunk, mask=block_start < D_MODEL_K)

class ChunkGateRecurrent:
    def __init__(self, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V):
        self.NUM_HEAD = NUM_HEAD
        self.NUM_BLOCK = NUM_BLOCK
        self.D_MODEL_K = D_MODEL_K
        self.D_MODEL_V = D_MODEL_V
        self.BLOCK_MODEL_K = BLOCK_MODEL_K
        self.BLOCK_MODEL_V = BLOCK_MODEL_V

    def forward(self, S, d, last_kv=None):
        O = torch.empty_like(S)
        grid = (self.NUM_HEAD * self.NUM_BLOCK,)
        _fwd_recurrence[grid](S, d, O, self.NUM_HEAD, self.NUM_BLOCK, self.D_MODEL_K, self.D_MODEL_V, self.BLOCK_MODEL_K, self.BLOCK_MODEL_V, last_kv)
        return O

    def backward(self, S, d, DI, DG, DL):
        DS = torch.empty_like(S)
        grid = (self.NUM_HEAD * self.NUM_BLOCK,)
        _bwd_recurrence[grid](S, d, DI, DG, DL, DS, self.NUM_HEAD, self.NUM_BLOCK, self.D_MODEL_K, self.D_MODEL_V, self.BLOCK_MODEL_K, self.BLOCK_MODEL_V)
        return DS
