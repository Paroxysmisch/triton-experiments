import triton
import triton.language as tl

# Triton kernel for forward recurrence
@triton.jit
def _fwd_recurrence(S, d, O, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, last_kv):
    pid = tl.program_id(0)
    num_blocks = NUM_HEAD * NUM_BLOCK
    num_elements = num_blocks * D_MODEL_K * D_MODEL_V

    # Load last_kv if provided
    if last_kv is not None:
        last_kv_ptr = last_kv + pid * D_MODEL_K * D_MODEL_V
        last_kv_block = tl.load(last_kv_ptr, mask=pid < num_blocks, other=0.0)
    else:
        last_kv_block = 0.0

    # Initialize accumulators
    accum = tl.zeros(D_MODEL_K * D_MODEL_V, dtype=tl.float32)

    # Iterate over blocks
    for block in range(NUM_BLOCK):
        block_idx = pid % NUM_BLOCK
        head_idx = pid // NUM_BLOCK

        # Load input data
        S_ptr = S + pid * D_MODEL_K * D_MODEL_V
        d_ptr = d + pid * D_MODEL_K * D_MODEL_V
        S_block = tl.load(S_ptr, mask=pid < num_blocks, other=0.0)
        d_block = tl.load(d_ptr, mask=pid < num_blocks, other=0.0)

        # Apply recurrent transformation
        accum = accum + S_block * d_block

        # Update accum with last_kv
        accum = accum + last_kv_block

        # Store result in output tensor
        O_ptr = O + pid * D_MODEL_K * D_MODEL_V
        tl.store(O_ptr, accum, mask=pid < num_blocks)

# Triton kernel for backward recurrence
@triton.jit
def _bwd_recurrence(S, d, DI, DG, DL, DS, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V):
    pid = tl.program_id(0)
    num_blocks = NUM_HEAD * NUM_BLOCK
    num_elements = num_blocks * D_MODEL_K * D_MODEL_V

    # Initialize accumulators
    accum_di = tl.zeros(D_MODEL_K * D_MODEL_V, dtype=tl.float32)
    accum_dg = tl.zeros(D_MODEL_K * D_MODEL_V, dtype=tl.float32)
    accum_dl = tl.zeros(D_MODEL_K * D_MODEL_V, dtype=tl.float32)

    # Iterate over blocks in reverse
    for block in range(NUM_BLOCK - 1, -1, -1):
        block_idx = pid % NUM_BLOCK
        head_idx = pid // NUM_BLOCK

        # Load input data
        S_ptr = S + pid * D_MODEL_K * D_MODEL_V
        d_ptr = d + pid * D_MODEL_K * D_MODEL_V
        S_block = tl.load(S_ptr, mask=pid < num_blocks, other=0.0)
        d_block = tl.load(d_ptr, mask=pid < num_blocks, other=0.0)

        # Load gradients
        DI_ptr = DI + pid * D_MODEL_K * D_MODEL_V
        DG_ptr = DG + pid * D_MODEL_K * D_MODEL_V
        DL_ptr = DL + pid * D_MODEL_K * D_MODEL_V
        DI_block = tl.load(DI_ptr, mask=pid < num_blocks, other=0.0)
        DG_block = tl.load(DG_ptr, mask=pid < num_blocks, other=0.0)
        DL_block = tl.load(DL_ptr, mask=pid < num_blocks, other=0.0)

        # Compute gradients
        accum_di = accum_di + DI_block
        accum_dg = accum_dg + DG_block
        accum_dl = accum_dl + DL_block

        # Accumulate gradients with respect to input data
        accum_di = accum_di + d_block * accum_dl
        accum_dg = accum_dg + S_block * accum_dl
        accum_dl = accum_dl + accum_di * d_block + accum_dg * S_block

        # Store gradients in output tensors
        DI_ptr = DI + pid * D_MODEL_K * D_MODEL_V
        DG_ptr = DG + pid * D_MODEL_K * D_MODEL_V
        DL_ptr = DL + pid * D_MODEL_K * D_MODEL_V
        tl.store(DI_ptr, accum_di, mask=pid < num_blocks)
        tl.store(DG_ptr, accum_dg, mask=pid < num_blocks)
        tl.store(DL_ptr, accum_dl, mask=pid < num_blocks)

# Custom autograd function for chunked gate recurrent
class ChunkGateRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cross_decay, last_kv=None):
        NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V = kv.shape[0], kv.shape[1], kv.shape[2], kv.shape[3]
        BLOCK_MODEL_K, BLOCK_MODEL_V = D_MODEL_K // NUM_HEAD, D_MODEL_V // NUM_HEAD

        O = torch.zeros_like(kv)
        _fwd_recurrence[triton.cdiv(O.numel(), BLOCK_MODEL_K * BLOCK_MODEL_V)](
            kv, cross_decay, O, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V, last_kv
        )

        ctx.save_for_backward(kv, cross_decay, last_kv, O)
        return O

    @staticmethod
    def backward(ctx, DO):
        kv, cross_decay, last_kv, O = ctx.saved_tensors
        NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V = kv.shape[0], kv.shape[1], kv.shape[2], kv.shape[3]
        BLOCK_MODEL_K, BLOCK_MODEL_V = D_MODEL_K // NUM_HEAD, D_MODEL_V // NUM_HEAD

        DI = torch.zeros_like(kv)
        DG = torch.zeros_like(kv)
        DL = torch.zeros_like(kv)
        DS = torch.zeros_like(kv)
        _bwd_recurrence[triton.cdiv(DI.numel(), BLOCK_MODEL_K * BLOCK_MODEL_V)](
            kv, cross_decay, DI, DG, DL, DS, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V
        )

        return DI, DG, DL, DS, None, None

# Example usage
if __name__ == "__main__":
    import torch

    # Example input tensors
    kv = torch.randn(2, 3, 4, 5, device="cuda")
    cross_decay = torch.randn(2, 3, 4, 5, device="cuda")

    # Forward pass
    output = ChunkGateRecurrent.apply(kv, cross_decay)

    # Backward pass
    output.sum().backward()
