import triton
import triton.language as tl
import torch

# Define Triton kernel for attention forward pass
@triton.jit
def _attn_fwd(q_ptr, k_ptr, v_ptr, o_ptr,
              q_scale, k_scale,
              stride_qm, stride_qn,
              stride_km, stride_kn,
              stride_vm, stride_vn,
              stride_om, stride_on,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr):
    # Program ID for grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute block offsets
    q_offset = pid_m * BLOCK_M
    k_offset = pid_n * BLOCK_N

    # Define pointers to blocks in Q, K, V, and output O
    q_block_ptr = q_ptr + q_offset * stride_qm
    k_block_ptr = k_ptr + k_offset * stride_kn
    v_block_ptr = v_ptr + k_offset * stride_vn
    o_block_ptr = o_ptr + q_offset * stride_om

    # Allocate shared memory for Q, K, V
    q_shared = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_shared = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
    v_shared = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)

    # Initialize accumulator for output
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load Q and scale it
    q = tl.load(q_block_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_qn + tl.arange(0, BLOCK_N)[None, :])
    q = q * q_scale

    # Process in stages
    for stage in range(STAGE):
        # Load K and V for the current stage
        k = tl.load(k_block_ptr + tl.arange(0, BLOCK_N)[:, None] * stride_kn + tl.arange(0, BLOCK_N)[None, :])
        v = tl.load(v_block_ptr + tl.arange(0, BLOCK_N)[:, None] * stride_vn + tl.arange(0, BLOCK_N)[None, :])

        # Scale K
        k = k * k_scale

        # Compute dot product between Q and K
        scores = tl.dot(q, k)

        # Apply softmax to scores
        max_scores = tl.max(scores, axis=1)
        scores = scores - max_scores[:, None]
        exp_scores = tl.exp(scores)
        sum_exp_scores = tl.sum(exp_scores, axis=1)
        softmax_scores = exp_scores / sum_exp_scores[:, None]

        # Update accumulator with weighted V
        acc += tl.dot(softmax_scores, v)

        # Update offsets for next stage
        k_block_ptr += BLOCK_N * stride_kn
        v_block_ptr += BLOCK_N * stride_vn

    # Write accumulated result to output
    tl.store(o_block_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_on + tl.arange(0, BLOCK_N)[None, :], acc)


# Define wrapper function
def forward(q, k, v, q_scale, k_scale):
    # Get input dimensions
    B, M, N = q.shape
    _, _, H = v.shape

    # Define block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    STAGE = 4  # Number of stages

    # Create output tensor
    o = torch.empty((B, M, H), device=q.device, dtype=q.dtype)

    # Launch Triton kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _attn_fwd[grid](
        q_ptr=q, k_ptr=k, v_ptr=v, o_ptr=o,
        q_scale=q_scale, k_scale=k_scale,
        stride_qm=q.stride(0), stride_qn=q.stride(1),
        stride_km=k.stride(0), stride_kn=k.stride(1),
        stride_vm=v.stride(0), stride_vn=v.stride(1),
        stride_om=o.stride(0), stride_on=o.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, STAGE=STAGE
    )

    return o
