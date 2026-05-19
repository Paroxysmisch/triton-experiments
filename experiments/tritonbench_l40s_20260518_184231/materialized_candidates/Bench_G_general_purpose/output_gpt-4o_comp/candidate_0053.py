import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_recurrence(S, d, O, last_kv, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, **meta):
    # Block and thread identifiers
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Calculate the starting index for this block
    start_k = block_id * BLOCK_MODEL_K
    start_v = block_id * BLOCK_MODEL_V

    # Initialize accumulators
    acc_k = tl.zeros((BLOCK_MODEL_K,), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_MODEL_V,), dtype=tl.float32)

    # Load previous state if available
    if last_kv is not None:
        acc_k = tl.load(last_kv + start_k)
        acc_v = tl.load(last_kv + start_v)

    # Iterate over blocks
    for b in range(NUM_BLOCK):
        # Load inputs for this block
        s = tl.load(S + b * BLOCK_MODEL_K)
        decay = tl.load(d + b)

        # Apply recurrent transformation
        acc_k = decay * acc_k + s
        acc_v = decay * acc_v + s

        # Store results in output tensor
        tl.store(O + start_k, acc_k)
        tl.store(O + start_v, acc_v)

@triton.jit
def _bwd_recurrence(S, d, DI, DG, DL, DS, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, **meta):
    # Block and thread identifiers
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Calculate the starting index for this block
    start_k = block_id * BLOCK_MODEL_K
    start_v = block_id * BLOCK_MODEL_V

    # Initialize gradient accumulators
    grad_k = tl.zeros((BLOCK_MODEL_K,), dtype=tl.float32)
    grad_v = tl.zeros((BLOCK_MODEL_V,), dtype=tl.float32)

    # Iterate over blocks in reverse
    for b in range(NUM_BLOCK - 1, -1, -1):
        # Load inputs for this block
        s = tl.load(S + b * BLOCK_MODEL_K)
        decay = tl.load(d + b)

        # Compute gradients
        grad_k = decay * grad_k + s
        grad_v = decay * grad_v + s

        # Store gradients in gradient tensors
        tl.store(DI + start_k, grad_k)
        tl.store(DG + start_v, grad_v)

class ChunkGateRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cross_decay, last_kv=None):
        # Define dimensions and block sizes
        NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V = kv.shape
        BLOCK_MODEL_K = D_MODEL_K // NUM_BLOCK
        BLOCK_MODEL_V = D_MODEL_V // NUM_BLOCK

        # Prepare output tensor
        O = torch.empty_like(kv)

        # Launch forward kernel
        grid = (NUM_BLOCK, NUM_HEAD)
        _fwd_recurrence[grid](
            S=kv, d=cross_decay, O=O, last_kv=last_kv,
            NUM_HEAD=NUM_HEAD, NUM_BLOCK=NUM_BLOCK,
            D_MODEL_K=D_MODEL_K, D_MODEL_V=D_MODEL_V,
            BLOCK_MODEL_K=BLOCK_MODEL_K, BLOCK_MODEL_V=BLOCK_MODEL_V
        )

        # Save context for backward pass
        ctx.save_for_backward(kv, cross_decay, O)
        ctx.NUM_HEAD = NUM_HEAD
        ctx.NUM_BLOCK = NUM_BLOCK
        ctx.D_MODEL_K = D_MODEL_K
        ctx.D_MODEL_V = D_MODEL_V
        ctx.BLOCK_MODEL_K = BLOCK_MODEL_K
        ctx.BLOCK_MODEL_V = BLOCK_MODEL_V

        return O

    @staticmethod
    def backward(ctx, DO):
        # Retrieve saved tensors
        kv, cross_decay, O = ctx.saved_tensors
        NUM_HEAD = ctx.NUM_HEAD
        NUM_BLOCK = ctx.NUM_BLOCK
        D_MODEL_K = ctx.D_MODEL_K
        D_MODEL_V = ctx.D_MODEL_V
        BLOCK_MODEL_K = ctx.BLOCK_MODEL_K
        BLOCK_MODEL_V = ctx.BLOCK_MODEL_V

        # Prepare gradient tensors
        DI = torch.empty_like(kv)
        DG = torch.empty_like(kv)
        DL = torch.empty_like(kv)
        DS = torch.empty_like(kv)

        # Launch backward kernel
        grid = (NUM_BLOCK, NUM_HEAD)
        _bwd_recurrence[grid](
            S=kv, d=cross_decay, DI=DI, DG=DG, DL=DL, DS=DS,
            NUM_HEAD=NUM_HEAD, NUM_BLOCK=NUM_BLOCK,
            D_MODEL_K=D_MODEL_K, D_MODEL_V=D_MODEL_V,
            BLOCK_MODEL_K=BLOCK_MODEL_K, BLOCK_MODEL_V=BLOCK_MODEL_V
        )

        # Return gradients
        return DI, DG, DL, DS

# Example usage
kv = torch.randn((4, 8, 64, 128), device='cuda')
cross_decay = torch.randn((8,), device='cuda')
chunk_gate_recurrent = ChunkGateRecurrent.apply
output = chunk_gate_recurrent(kv, cross_decay)
