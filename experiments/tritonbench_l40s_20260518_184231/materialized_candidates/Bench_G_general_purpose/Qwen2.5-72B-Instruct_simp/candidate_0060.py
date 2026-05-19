import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    q_ptr, k_ptr, cos_ptr, sin_ptr, q_out_ptr, k_out_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_feat_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_feat_stride,
    cos_seq_stride, cos_feat_stride,
    sin_seq_stride, sin_feat_stride,
    q_out_batch_stride, q_out_head_stride, q_out_seq_stride, q_out_feat_stride,
    k_out_batch_stride, k_out_head_stride, k_out_seq_stride, k_out_feat_stride,
    N_CTX, HEAD_DIM, BACKWARD_PASS: tl.constexpr
):
    # Compute the block ID and thread ID
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)
    block_size = N_CTX // num_blocks

    # Compute the start and end indices for the current block
    start = pid * block_size
    end = (pid + 1) * block_size if pid < num_blocks - 1 else N_CTX

    # Compute the batch, head, and feature indices
    batch_id = tl.program_id(1)
    head_id = tl.program_id(2)

    # Load the query and key matrices
    q = tl.load(q_ptr + batch_id * q_batch_stride + head_id * q_head_stride + tl.arange(0, block_size) * q_seq_stride + tl.arange(0, HEAD_DIM) * q_feat_stride, mask=start + tl.arange(0, block_size) < N_CTX, other=0.0)
    k = tl.load(k_ptr + batch_id * k_batch_stride + head_id * k_head_stride + tl.arange(0, block_size) * k_seq_stride + tl.arange(0, HEAD_DIM) * k_feat_stride, mask=start + tl.arange(0, block_size) < N_CTX, other=0.0)

    # Load the cosine and sine values
    cos = tl.load(cos_ptr + tl.arange(0, block_size) * cos_seq_stride + tl.arange(0, HEAD_DIM) * cos_feat_stride, mask=start + tl.arange(0, block_size) < N_CTX, other=0.0)
    sin = tl.load(sin_ptr + tl.arange(0, block_size) * sin_seq_stride + tl.arange(0, HEAD_DIM) * sin_feat_stride, mask=start + tl.arange(0, block_size) < N_CTX, other=0.0)

    # Apply the RoPE transformation
    q_rot = q * cos - tl.flip(q, 1) * sin
    k_rot = k * cos - tl.flip(k, 1) * sin

    # Store the results
    tl.store(q_out_ptr + batch_id * q_out_batch_stride + head_id * q_out_head_stride + tl.arange(0, block_size) * q_out_seq_stride + tl.arange(0, HEAD_DIM) * q_out_feat_stride, q_rot, mask=start + tl.arange(0, block_size) < N_CTX)
    tl.store(k_out_ptr + batch_id * k_out_batch_stride + head_id * k_out_head_stride + tl.arange(0, block_size) * k_out_seq_stride + tl.arange(0, HEAD_DIM) * k_out_feat_stride, k_rot, mask=start + tl.arange(0, block_size) < N_CTX)

    # If this is a backward pass, transpose the inputs and outputs
    if BACKWARD_PASS:
        q_rot = tl.transpose(q_rot)
        k_rot = tl.transpose(k_rot)

        tl.store(q_out_ptr + batch_id * q_out_batch_stride + head_id * q_out_head_stride + tl.arange(0, block_size) * q_out_seq_stride + tl.arange(0, HEAD_DIM) * q_out_feat_stride, q_rot, mask=start + tl.arange(0, block_SIZE) < N_CTX)
        tl.store(k_out_ptr + batch_id * k_out_batch_stride + head_id * k_out_head_stride + tl.arange(0, block_size) * k_out_seq_stride + tl.arange(0, HEAD_DIM) * k_out_feat_stride, k_rot, mask=start + tl.arange(0, block_size) < N_CTX)

import torch

def rope_backward(q_grad, k_grad, cos, sin, q_shape, k_shape, N_CTX, HEAD_DIM):
    # Transpose the input gradients
    q_grad = q_grad.transpose(1, 2).contiguous()
    k_grad = k_grad.transpose(1, 2).contiguous()

    # Allocate output tensors
    q_out_grad = torch.empty_like(q_grad)
    k_out_grad = torch.empty_like(k_grad)

    # Define the grid and block dimensions
    grid = (N_CTX // 32, q_shape[0], q_shape[1])
    block = (32, HEAD_DIM)

    # Launch the kernel
    _triton_rope[grid, block](
        q_grad, k_grad, cos, sin, q_out_grad, k_out_grad,
        q_grad.stride(0), q_grad.stride(1), q_grad.stride(2), q_grad.stride(3),
        k_grad.stride(0), k_grad.stride(1), k_grad.stride(2), k_grad.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        q_out_grad.stride(0), q_out_grad.stride(1), q_out_grad.stride(2), q_out_grad.stride(3),
        k_out_grad.stride(0), k_out_grad.stride(1), k_out_grad.stride(2), k_out_grad.stride(3),
        N_CTX, HEAD_DIM, True
    )

    # Transpose the output gradients back to the original shape
    q_out_grad = q_out_grad.transpose(1, 2).contiguous()
    k_out_grad = k_out_grad.transpose(1, 2).contiguous()

    return q_out_grad, k_out_grad
