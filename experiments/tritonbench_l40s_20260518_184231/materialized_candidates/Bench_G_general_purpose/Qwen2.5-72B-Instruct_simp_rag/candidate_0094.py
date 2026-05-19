import triton
import triton.language as tl

@triton.jit
def _rotary_kernel(
    Q, K, Cos, Sin,
    stride_q, stride_k, stride_c, stride_s,
    max_total_len, HEAD_Q, HEAD_K,
    BLOCK_HEAD: tl.constexpr, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Compute the block indices
    head_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    # Compute the base pointers for Q, K, Cos, and Sin
    q_ptr = Q + head_idx * stride_q + seq_idx * BLOCK_SEQ * BLOCK_DMODEL
    k_ptr = K + head_idx * stride_k + seq_idx * BLOCK_SEQ * BLOCK_DMODEL
    cos_ptr = Cos + seq_idx * BLOCK_SEQ * BLOCK_DMODEL
    sin_ptr = Sin + seq_idx * BLOCK_SEQ * BLOCK_DMODEL

    # Load the slices from Q and K
    q = tl.load(q_ptr, mask=seq_idx < max_total_len, other=0.0)
    k = tl.load(k_ptr, mask=seq_idx < max_total_len, other=0.0)
    cos = tl.load(cos_ptr, mask=seq_idx < max_total_len, other=1.0)
    sin = tl.load(sin_ptr, mask=seq_idx < max_total_len, other=0.0)

    # Apply the rotary transformation
    q_rotated = q * cos - k * sin
    k_rotated = k * cos + q * sin

    # Store the results back into Q and K
    tl.store(q_ptr, q_rotated, mask=seq_idx < max_total_len)
    tl.store(k_ptr, k_rotated, mask=seq_idx < max_total_len)

import torch

def rotary_emb_fwd(
    Q, K, Cos, Sin,
    stride_q, stride_k, stride_c, stride_s,
    max_total_len, HEAD_Q, HEAD_K,
    BLOCK_HEAD=16, BLOCK_SEQ=16, BLOCK_DMODEL=64
):
    # Determine the grid size for execution
    grid = (HEAD_Q, (max_total_len + BLOCK_SEQ - 1) // BLOCK_SEQ, 1)

    # Determine the number of warps based on the dimension size
    num_warps = 4 if BLOCK_DMODEL <= 256 else 8

    # Launch the kernel
    _rotary_kernel[grid](
        Q, K, Cos, Sin,
        stride_q, stride_k, stride_c, stride_s,
        max_total_len, HEAD_Q, HEAD_K,
        BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL,
        num_warps=num_warps
    )
