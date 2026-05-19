import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    sm_scale: tl.float32, 
    M: tl.int32, N: tl.int32, H: tl.int32,
    IS_CAUSAL: tl.constexpr, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    USE_FP8: tl.constexpr
):
    pid = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    block_m = tl.program_id(axis=2)

    # Compute the row and column indices for the current program instance
    row_start = pid * BLOCK_M + block_m * BLOCK_M
    col_start = head_id * N

    # Compute the block boundaries
    row_end = tl.minimum(row_start + BLOCK_M, M)
    col_end = tl.minimum(col_start + BLOCK_N, N)

    # Initialize the output block
    out_block = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    # Load the query block
    q_block = tl.load(Q + (row_start * BLOCK_DMODEL + head_id * BLOCK_DMODEL * M + tl.arange(0, BLOCK_DMODEL))[:, None], mask=row_start + tl.arange(0, BLOCK_M) < M, other=0.0)

    # Iterate over the key and value blocks
    for n in range(0, N, BLOCK_N):
        k_block = tl.load(K + (n * BLOCK_DMODEL + head_id * BLOCK_DMODEL * M + tl.arange(0, BLOCK_DMODEL))[:, None], mask=n + tl.arange(0, BLOCK_N) < N, other=0.0)
        v_block = tl.load(V + (n * BLOCK_DMODEL + head_id * BLOCK_DMODEL * M + tl.arange(0, BLOCK_DMODEL))[:, None], mask=n + tl.arange(0, BLOCK_N) < N, other=0.0)

        # Compute the attention scores
        scores = tl.dot(q_block, k_block, trans_b=True) * sm_scale

        # Apply causal masking if required
        if IS_CAUSAL:
            mask = tl.arange(0, BLOCK_N)[None, :] < (row_start + tl.arange(0, BLOCK_M))[:, None]
            scores = tl.where(mask, scores, float('-inf'))

        # Apply softmax
        lse = tl.max(scores, 1)
        scores = tl.exp(scores - lse[:, None])
        lse = tl.sum(scores, 1)
        scores = scores / lse[:, None]

        # Compute the output block
        out_block += tl.dot(scores, v_block)

    # Store the output block
    tl.store(Out + (row_start * BLOCK_DMODEL + head_id * BLOCK_DMODEL * M + tl.arange(0, BLOCK_DMODEL))[:, None], out_block, mask=row_start + tl.arange(0, BLOCK_M) < M)

import torch
import triton
import triton.language as tl

def triton_fa(Q, K, V, sm_scale, is_causal, use_fp8=False):
    # Check input shapes and data types
    assert Q.shape == K.shape == V.shape, "Q, K, and V must have the same shape"
    assert Q.dtype == K.dtype == V.dtype, "Q, K, and V must have the same data type"
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Q, K, and V must be contiguous"

    # Get input dimensions
    M, N, H, D = Q.shape
    BLOCK_M, BLOCK_N, BLOCK_DMODEL = 128, 128, D

    # Determine the grid size
    grid = (
        triton.cdiv(M, BLOCK_M),
        H,
        triton.cdiv(M, BLOCK_M)
    )

    # Determine the number of warps
    Lk = BLOCK_DMODEL // 32
    num_warps = 4 if Lk <= 64 else 8

    # Allocate output tensor
    Out = torch.empty_like(Q)

    # Launch the kernel
    _fwd_kernel[grid](
        Q, K, V, Out,
        sm_scale,
        M, N, H,
        is_causal,
        BLOCK_M, BLOCK_N, BLOCK_DMODEL,
        use_fp8
    )

    return Out
