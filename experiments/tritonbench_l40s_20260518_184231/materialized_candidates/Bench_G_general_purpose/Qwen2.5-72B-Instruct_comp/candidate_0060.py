import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    q_ptr, k_ptr, cos, sin, q_batch_stride, q_head_stride, q_seq_stride, q_head_dim_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_head_dim_stride, cos_seq_stride, cos_head_dim_stride,
    sin_seq_stride, sin_head_dim_stride, N_CTX, HEAD_DIM, BACKWARD_PASS: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // (N_CTX * HEAD_DIM)
    seq_id = (pid % (N_CTX * HEAD_DIM)) // HEAD_DIM
    head_id = (pid % HEAD_DIM)

    q_offset = (batch_id * q_batch_stride + seq_id * q_seq_stride + head_id * q_head_dim_stride)
    k_offset = (batch_id * k_batch_stride + seq_id * k_seq_stride + head_id * k_head_dim_stride)
    cos_offset = (seq_id * cos_seq_stride + head_id * cos_head_dim_stride)
    sin_offset = (seq_id * sin_seq_stride + head_id * sin_head_dim_stride)

    q = tl.load(q_ptr + q_offset)
    k = tl.load(k_ptr + k_offset)
    cos_val = tl.load(cos + cos_offset)
    sin_val = tl.load(sin + sin_offset)

    half_dim = HEAD_DIM // 2

    q1 = q[:half_dim]
    q2 = q[half_dim:]
    k1 = k[:half_dim]
    k2 = k[half_dim:]

    if BACKWARD_PASS:
        q_rotated1 = q1 * cos_val + q2 * sin_val
        q_rotated2 = -q2 * cos_val + q1 * sin_val
        k_rotated1 = k1 * cos_val + k2 * sin_val
        k_rotated2 = -k2 * cos_val + k1 * sin_val
    else:
        q_rotated1 = q1 * cos_val - q2 * sin_val
        q_rotated2 = q2 * cos_val + q1 * sin_val
        k_rotated1 = k1 * cos_val - k2 * sin_val
        k_rotated2 = k2 * cos_val + k1 * sin_val

    q_rotated = tl.concat([q_rotated1, q_rotated2])
    k_rotated = tl.concat([k_rotated1, k_rotated2])

    tl.store(q_ptr + q_offset, q_rotated)
    tl.store(k_ptr + k_offset, k_rotated)

import triton
import triton.language as tl

def rope_backward(dq, dk, cos, sin, q_batch_stride, q_head_stride, q_seq_stride, q_head_dim_stride,
                  k_batch_stride, k_head_stride, k_seq_stride, k_head_dim_stride, cos_seq_stride, cos_head_dim_stride,
                  sin_seq_stride, sin_head_dim_stride, N_CTX, HEAD_DIM, BLOCK_SIZE_SEQ, BLOCK_SIZE_HEAD_DIM, BACKWARD_PASS):
    # Ensure inputs are padded to power-of-two dimensions
    N_CTX = (N_CTX + BLOCK_SIZE_SEQ - 1) // BLOCK_SIZE_SEQ * BLOCK_SIZE_SEQ
    HEAD_DIM = (HEAD_DIM + BLOCK_SIZE_HEAD_DIM - 1) // BLOCK_SIZE_HEAD_DIM * BLOCK_SIZE_HEAD_DIM

    # Launch the kernel
    grid = (N_CTX * HEAD_DIM, 1, 1)
    _triton_rope[grid](
        dq, dk, cos, sin, q_batch_stride, q_head_stride, q_seq_stride, q_head_dim_stride,
        k_batch_stride, k_head_stride, k_seq_stride, k_head_dim_stride, cos_seq_stride, cos_head_dim_stride,
        sin_seq_stride, sin_head_dim_stride, N_CTX, HEAD_DIM, BACKWARD_PASS
    )
