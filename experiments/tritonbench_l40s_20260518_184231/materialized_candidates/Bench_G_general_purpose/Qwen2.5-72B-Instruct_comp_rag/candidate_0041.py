import triton
import triton.language as tl

BACKWARD_PASS = False  # Set to True for backward pass

@triton.jit
def _triton_rope(
    Q,  # Pointers to matrices
    K,
    COS,
    SIN,
    # Matrix dimensions
    batch_size,
    seq_len,
    n_heads,
    head_dim,
    rotary_dim,
    # Strides
    stride_q_batch,
    stride_q_seq,
    stride_q_heads,
    stride_q_dim,
    stride_k_batch,
    stride_k_seq,
    stride_k_heads,
    stride_k_dim,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BACKWARD_PASS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(seq_len, BLOCK_M)
    num_pid_n = tl.cdiv(head_dim, BLOCK_N)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid %= num_pid_in_batch
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_b = batch_id * stride_q_batch

    q = Q + offset_b + offset_m[:, None] * stride_q_seq + offset_n[None, :] * stride_q_dim
    k = K + offset_b + offset_m[:, None] * stride_k_seq + offset_n[None, :] * stride_k_dim

    cos = COS + offset_m[:, None] * rotary_dim + offset_n[None, :] % rotary_dim
    sin = SIN + offset_m[:, None] * rotary_dim + offset_n[None, :] % rotary_dim

    q_rot = tl.load(q, mask=offset_m[:, None] < seq_len, other=0.0).to(tl.float32)
    k_rot = tl.load(k, mask=offset_m[:, None] < seq_len, other=0.0).to(tl.float32)

    cos = tl.load(cos, mask=offset_m[:, None] < seq_len, other=1.0).to(tl.float32)
    sin = tl.load(sin, mask=offset_m[:, None] < seq_len, other=0.0).to(tl.float32)

    q0 = q_rot[:, :rotary_dim // 2]
    q1 = q_rot[:, rotary_dim // 2:]
    k0 = k_rot[:, :rotary_dim // 2]
    k1 = k_rot[:, rotary_dim // 2:]

    if BACKWARD_PASS:
        sin = -sin

    q_rot_new = tl.stack(q0 * cos - q1 * sin, q0 * sin + q1 * cos)
    k_rot_new = tl.stack(k0 * cos - k1 * sin, k0 * sin + k1 * cos)

    q_rot_new = q_rot_new.reshape((BLOCK_M, head_dim))
    k_rot_new = k_rot_new.reshape((BLOCK_M, head_dim))

    tl.store(q, q_rot_new, mask=offset_m[:, None] < seq_len)
    tl.store(k, k_rot_new, mask=offset_m[:, None] < seq_len)

import torch

def rope_forward(q, k, cos, sin, batch_size, seq_len, n_heads, head_dim, rotary_dim):
    assert q.shape == (batch_size, seq_len, n_heads, head_dim)
    assert k.shape == (batch_size, seq_len, n_heads, head_dim)
    assert cos.shape == (seq_len, rotary_dim)
    assert sin.shape == (seq_len, rotary_dim)

    # Transpose the query and key matrices to the appropriate format
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()

    # Compute the necessary paddings using Triton's utility functions
    BLOCK_M = 16
    BLOCK_N = 32

    # Ensure the inputs are contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    # Configure the execution grid
    grid = (triton.cdiv(seq_len, BLOCK_M) * triton.cdiv(head_dim, BLOCK_N) * batch_size * n_heads,)

    # Launch the kernel
    _triton_rope[grid](
        q,  # Pointers to matrices
        k,
        cos,
        sin,
        batch_size,  # Matrix dimensions
        seq_len,
        n_heads,
        head_dim,
        rotary_dim,
        # Strides
        q.stride(0),
        q.stride(2),
        q.stride(1),
        q.stride(3),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        k.stride(3),
        # Meta-parameters
        BLOCK_M,
        BLOCK_N,
        BACKWARD_PASS
    )

    # Return the matrices to their original shapes
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)

    return q, k, cos, sin
