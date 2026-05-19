import triton
import triton.language as tl

@triton.jit
def _rotary_kernel(
    Q, K, Cos, Sin,
    stride_q, stride_k, stride_cos, stride_sin,
    stride_q_head, stride_k_head, stride_cos_head, stride_sin_head,
    stride_q_seq, stride_k_seq, stride_cos_seq, stride_sin_seq,
    stride_q_d, stride_k_d, stride_cos_d, stride_sin_d,
    BLOCK_HEAD: tl.constexpr, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Get the current head and sequence indices
    head_id = tl.program_id(0)
    seq_id = tl.program_id(1)

    # Calculate the offsets for Q and K
    q_offset = head_id * stride_q_head + seq_id * stride_q_seq
    k_offset = head_id * stride_k_head + seq_id * stride_k_seq
    cos_offset = head_id * stride_cos_head + seq_id * stride_cos_seq
    sin_offset = head_id * stride_sin_head + seq_id * stride_sin_seq

    # Load the segments of Q and K
    q = tl.load(Q + q_offset, mask=seq_id < BLOCK_SEQ, other=0.0)
    k = tl.load(K + k_offset, mask=seq_id < BLOCK_SEQ, other=0.0)
    cos = tl.load(Cos + cos_offset, mask=seq_id < BLOCK_SEQ, other=1.0)
    sin = tl.load(Sin + sin_offset, mask=seq_id < BLOCK_SEQ, other=0.0)

    # Apply the rotary transformation
    q0 = q[:, 0::2]
    q1 = q[:, 1::2]
    k0 = k[:, 0::2]
    k1 = k[:, 1::2]

    out0 = q0 * cos - q1 * sin
    out1 = q0 * sin + q1 * cos
    out0_k = k0 * cos - k1 * sin
    out1_k = k0 * sin + k1 * cos

    # Store the transformed segments back to the original locations in Q and K
    tl.store(Q + q_offset, out0, mask=seq_id < BLOCK_SEQ)
    tl.store(Q + q_offset + 1, out1, mask=seq_id < BLOCK_SEQ)
    tl.store(K + k_offset, out0_k, mask=seq_id < BLOCK_SEQ)
    tl.store(K + k_offset + 1, out1_k, mask=seq_id < BLOCK_SEQ)

import torch

def rotary_emb_fwd(Q, K, Cos, Sin, BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL):
    # Validate input shapes
    assert Q.shape == K.shape, "Q and K must have the same shape"
    assert Q.shape[0] == Cos.shape[0] == Sin.shape[0], "Q, Cos, and Sin must have the same number of heads"
    assert Q.shape[1] == Cos.shape[1] == Sin.shape[1], "Q, Cos, and Sin must have the same sequence length"
    assert Q.shape[2] == Cos.shape[2] == Sin.shape[2], "Q, Cos, and Sin must have the same head dimension"

    # Calculate strides
    stride_q = Q.stride(0) * Q.element_size()
    stride_k = K.stride(0) * K.element_size()
    stride_cos = Cos.stride(0) * Cos.element_size()
    stride_sin = Sin.stride(0) * Sin.element_size()

    stride_q_head = Q.stride(1) * Q.element_size()
    stride_k_head = K.stride(1) * K.element_size()
    stride_cos_head = Cos.stride(1) * Cos.element_size()
    stride_sin_head = Sin.stride(1) * Sin.element_size()

    stride_q_seq = Q.stride(2) * Q.element_size()
    stride_k_seq = K.stride(2) * K.element_size()
    stride_cos_seq = Cos.stride(2) * Cos.element_size()
    stride_sin_seq = Sin.stride(2) * Sin.element_size()

    stride_q_d = Q.stride(3) * Q.element_size()
    stride_k_d = K.stride(3) * K.element_size()
    stride_cos_d = Cos.stride(3) * Cos.element_size()
    stride_sin_d = Sin.stride(3) * Sin.element_size()

    # Calculate grid dimensions
    grid = (Q.shape[0], Q.shape[1])

    # Determine the number of warps
    num_warps = BLOCK_DMODEL // 32

    # Launch the kernel
    _rotary_kernel[grid](
        Q, K, Cos, Sin,
        stride_q, stride_k, stride_cos, stride_sin,
        stride_q_head, stride_k_head, stride_cos_head, stride_sin_head,
        stride_q_seq, stride_k_seq, stride_cos_seq, stride_sin_seq,
        stride_q_d, stride_k_d, stride_cos_d, stride_sin_d,
        BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL,
        num_warps=num_warps
    )

    return Q, K
