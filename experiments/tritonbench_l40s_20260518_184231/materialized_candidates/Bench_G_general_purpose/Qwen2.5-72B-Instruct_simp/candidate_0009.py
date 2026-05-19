import triton
import triton.language as tl

@triton.jit
def _rope_embedding(
    Q,  # Pointer to the input tensor
    cos,  # Pointer to the cosine tensor
    sin,  # Pointer to the sine tensor
    Q_out,  # Pointer to the output tensor
    stride_qb, stride_qh, stride_qd,  # Strides for Q
    stride_cosb, stride_cosd,  # Strides for cos
    stride_sinb, stride_sind,  # Strides for sin
    stride_q_outb, stride_q_outh, stride_q_outd,  # Strides for Q_out
    n_elements,  # Number of elements in the batch
    n_heads,  # Number of heads
    head_dim,  # Dimension of each head
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    batch_start = pid * BLOCK_SIZE
    batch_end = min(batch_start + BLOCK_SIZE, n_elements)

    for b in range(batch_start, batch_end):
        for h in range(n_heads):
            for d in range(head_dim):
                q_val = tl.load(Q + b * stride_qb + h * stride_qh + d * stride_qd)
                cos_val = tl.load(cos + b * stride_cosb + d * stride_cosd)
                sin_val = tl.load(sin + b * stride_sinb + d * stride_sind)

                # Split and rotate half
                q1 = q_val[:head_dim // 2]
                q2 = q_val[head_dim // 2:]
                q2_rotated = tl.concat([q2[1:], q2[:1]])

                # Apply RoPE transformation
                q_rotated = q1 * cos_val + q2_rotated * sin_val

                # Store the result
                tl.store(Q_out + b * stride_q_outb + h * stride_q_outh + d * stride_q_outd, q_rotated)

import torch
import triton
import triton.language as tl

def calculate_settings(Q, cos, sin, BLOCK_SIZE):
    n_elements, n_heads, head_dim = Q.shape
    grid = (n_elements // BLOCK_SIZE + (n_elements % BLOCK_SIZE > 0),)
    return grid, n_heads, head_dim, BLOCK_SIZE

def _rope_embedding_forward_impl(Q, cos, sin, BLOCK_SIZE, num_warps=4):
    n_elements, n_heads, head_dim = Q.shape
    Q_out = torch.empty_like(Q)

    grid, n_heads, head_dim, BLOCK_SIZE = calculate_settings(Q, cos, sin, BLOCK_SIZE)

    _rope_embedding[grid](
        Q, cos, sin, Q_out,
        Q.stride(0), Q.stride(1), Q.stride(2),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        Q_out.stride(0), Q_out.stride(1), Q_out.stride(2),
        n_elements, n_heads, head_dim, BLOCK_SIZE,
        num_warps=num_warps
    )

    return Q_out

def _rope_embedding_backward_impl(dQ, cos, sin, BLOCK_SIZE, num_warps=4):
    n_elements, n_heads, head_dim = dQ.shape
    dQ_out = torch.empty_like(dQ)

    grid, n_heads, head_dim, BLOCK_SIZE = calculate_settings(dQ, cos, sin, BLOCK_SIZE)

    _rope_embedding[grid](
        dQ, cos, sin, dQ_out,
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        dQ_out.stride(0), dQ_out.stride(1), dQ_out.stride(2),
        n_elements, n_heads, head_dim, BLOCK_SIZE,
        num_warps=num_warps
    )

    return dQ_out
