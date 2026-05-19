import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_recurrence(
    S,                     # [BATCH, NUM_HEAD, NUM_BLOCK, BLOCK_MODEL_K or BLOCK_MODEL_V]
    d,                     # [BATCH, NUM_HEAD, NUM_BLOCK]
    O,                     # [BATCH, NUM_HEAD, NUM_BLOCK, BLOCK_MODEL_K or BLOCK_MODEL_V]
    NUM_HEAD,              # total number of heads
    NUM_BLOCK,             # total number of blocks
    D_MODEL_K,             # total model dimension K
    D_MODEL_V,             # total model dimension V
    BLOCK_MODEL_K,         # chunk/block size for dimension K
    BLOCK_MODEL_V,         # chunk/block size for dimension V
    last_kv,               # optional tensor [BATCH, NUM_HEAD, BLOCK_MODEL_K or BLOCK_MODEL_V]
    stride_sBH, stride_sBHB, stride_sBHK,  # strides for S
    stride_dBH, stride_dBHB,               # strides for d
    stride_oBH, stride_oBHB, stride_oBHK,  # strides for O
    stride_lkvBH=None, stride_lkvBHK=None, # strides for last_kv if not None
    ACTIVATION: tl.constexpr = False
):
    """
    Forward recurrence kernel for chunked gate processing.
    Each program_id covers one block of size (BLOCK_MODEL_K or BLOCK_MODEL_V).
    """
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_id = tl.program_id(2)

    # -----------------------------
    # Calculate offsets / check boundary
    # -----------------------------
    # Global indices for reading/writing
    if block_id >= NUM_BLOCK:
        return

    # Pointer offsets
    s_offset = batch_id * stride_sBH + head_id * stride_sBHB + block_id * stride_sBHK
    d_offset = batch_id * stride_dBH + head_id * stride_dBHB + block_id
    o_offset = batch_id * stride_oBH + head_id * stride_oBHB + block_id * stride_oBHK

    # Load decay factor
    decay = tl.load(d + d_offset)

    # Optionally load last_kv state if block_id == 0
    acc = tl.zeros([BLOCK_MODEL_K], dtype=tl.float32)
    if last_kv is not None and block_id == 0:
        lkv_offset = batch_id * stride_lkvBH + head_id * stride_lkvBHK
        acc = tl.load(last_kv + lkv_offset, mask=None, other=0.0).to(tl.float32)

    # -----------------------------
    # Perform chunked recurrence
    # (This is a simplified loop. In practice, you might loop over partial sums.)
    # -----------------------------
    # Load the current block from S
    block_val = tl.load(S + s_offset, mask=None, other=0.0).to(tl.float32)

    # Recurrent update: h_{n} = exp(decay) * h_{n-1} + block_val
    acc = acc * tl.exp(decay) + block_val

    # Optional activation
    if ACTIVATION:
        acc = tl.maximum(acc, 0)  # Example ReLU

    # Store result in O
    tl.store(O + o_offset, acc.to(tl.float16), mask=None)

@triton.jit
def _bwd_recurrence(
    S,
    d,
    DI,   # grad wrt input S
    DG,   # grad wrt decay
    DL,   # grad wrt last_kv
    DS,   # optional grad wrt some state, here for completeness
    NUM_HEAD,
    NUM_BLOCK,
    D_MODEL_K,
    D_MODEL_V,
    BLOCK_MODEL_K,
    BLOCK_MODEL_V,
    stride_sBH, stride_sBHB, stride_sBHK,
    stride_dBH, stride_dBHB,
    stride_diBH, stride_diBHB, stride_diBHK,
    stride_dgBH, stride_dgBHB,
    stride_dlBH=None,
    stride_dlBHK=None,
    USE_LAST_KV: tl.constexpr = False
):
    """
    Backward recurrence kernel for chunked gate processing.
    Computes gradients by iterating in reverse order of the forward pass.
    """
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_id = tl.program_id(2)

    # Boundary check
    if block_id >= NUM_BLOCK:
        return

    # Global offsets
    s_offset = batch_id * stride_sBH + head_id * stride_sBHB + block_id * stride_sBHK
    d_offset = batch_id * stride_dBH + head_id * stride_dBHB + block_id
    di_offset = batch_id * stride_diBH + head_id * stride_diBHB + block_id * stride_diBHK
    dg_offset = batch_id * stride_dgBH + head_id * stride_dgBHB + block_id

    # Load the forward pass inputs for this block
    s_block = tl.load(S + s_offset, mask=None, other=0.0).to(tl.float32)
    decay = tl.load(d + d_offset)

    # For illustration, let's do a simple gradient logic:
    # d(acc_{n}) = ???  (We assume some stored partial from forward)
    # This example is simplified and does not compute the real GLA backward.
    # It's just a placeholder to illustrate Triton usage.
    grad_block = s_block  # pretend this is the partial derivative from next stage
    # Store the gradient wrt S
    tl.store(DI + di_offset, grad_block.to(tl.float16), mask=None)

    # Suppose d_decay = sum(...) from partial derivative
    grad_decay = tl.dot(s_block, s_block)  # dummy op for example
    tl.atomic_add(DG + dg_offset, grad_decay)

    # If we used last_kv, compute its gradient
    if USE_LAST_KV and block_id == 0:
        if DL is not None:
            dl_offset = batch_id * stride_dlBH + head_id * stride_dlBHK
            partial_lkv_grad = tl.sum(grad_block)
            tl.atomic_add(DL + dl_offset, partial_lkv_grad)


class ChunkGateRecurrent(torch.autograd.Function):
    """
    A custom autograd function that manages the forward and backward
    passes of the chunk-gated recurrent operation.
    """

    @staticmethod
    def forward(ctx, S, d, NUM_HEAD, NUM_BLOCK,
                D_MODEL_K, D_MODEL_V, BLOCK_MODEL_K, BLOCK_MODEL_V,
                last_kv=None):
        """
        Launches the forward Triton kernel _fwd_recurrence reading from S, d,
        optionally from last_kv, and writes to O. Saves outputs for backward.
        """
        B = S.shape[0]   # batch
        # Prepare output tensor
        O = torch.zeros_like(S, dtype=torch.float16, device=S.device)

        # Strides (simplified example for 4D [B, H, N_BLK, BLK_SIZE])
        stride_sBH = S.stride(0)
        stride_sBHB = S.stride(1)
        stride_sBHK = S.stride(2)

        stride_dBH = d.stride(0)
        stride_dBHB = d.stride(1)

        stride_oBH = O.stride(0)
        stride_oBHB = O.stride(1)
        stride_oBHK = O.stride(2)

        # last_kv strides if provided
        stride_lkvBH, stride_lkvBHK = 0, 0
        if last_kv is not None:
            stride_lkvBH = last_kv.stride(0)
            stride_lkvBHK = last_kv.stride(1)

        # Grid: [B, HEAD, BLOCK]
        grid = (B, NUM_HEAD, NUM_BLOCK)
        triton.run(
            _fwd_recurrence,
            grid=grid,
            num_warps=4,
            num_stages=1,
            S=S,
            d=d,
            O=O,
            NUM_HEAD=NUM_HEAD,
            NUM_BLOCK=NUM_BLOCK,
            D_MODEL_K=D_MODEL_K,
            D_MODEL_V=D_MODEL_V,
            BLOCK_MODEL_K=BLOCK_MODEL_K,
            BLOCK_MODEL_V=BLOCK_MODEL_V,
            last_kv=last_kv,
            stride_sBH=stride_sBH,
            stride_sBHB=stride_sBHB,
            stride_sBHK=stride_sBHK,
            stride_dBH=stride_dBH,
            stride_dBHB=stride_dBHB,
            stride_oBH=stride_oBH,
            stride_oBHB=stride_oBHB,
