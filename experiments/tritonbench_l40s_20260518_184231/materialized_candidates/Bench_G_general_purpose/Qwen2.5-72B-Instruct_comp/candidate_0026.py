import triton
import triton.language as tl

@triton.jit
def q_kernel_per_block_int8(
    q_ptr, q_int8_ptr, q_scale_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_k_stride,
    q_int8_batch_stride, q_int8_head_stride, q_int8_seq_stride, q_int8_k_stride,
    q_scale_batch_stride, q_scale_head_stride,
    BLKQ: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    block_id = pid % num_blocks

    batch_id = block_id // (q_head_stride // q_k_stride)
    head_id = (block_id % (q_head_stride // q_k_stride)) // (q_seq_stride // q_k_stride)
    seq_id = block_id % (q_seq_stride // q_k_stride)

    q_offset = batch_id * q_batch_stride + head_id * q_head_stride + seq_id * q_seq_stride
    q_int8_offset = batch_id * q_int8_batch_stride + head_id * q_int8_head_stride + seq_id * q_int8_seq_stride
    q_scale_offset = batch_id * q_scale_batch_stride + head_id * q_scale_head_stride

    q_block = tl.load(q_ptr + q_offset + tl.arange(0, BLKQ), mask=tl.arange(0, BLKQ) < BLKQ, other=0.0)
    max_abs = tl.max(tl.abs(q_block), axis=0)
    scale = 127.0 / max_abs
    q_int8_block = tl.round(q_block * scale)
    q_int8_block = tl.to_int8(q_int8_block)

    tl.store(q_int8_ptr + q_int8_offset + tl.arange(0, BLKQ), q_int8_block, mask=tl.arange(0, BLKQ) < BLKQ)
    tl.store(q_scale_ptr + q_scale_offset, scale)

@triton.jit
def k_kernel_per_block_int8(
    k_ptr, k_int8_ptr, k_scale_ptr,
    k_batch_stride, k_head_stride, k_seq_stride, k_k_stride,
    k_int8_batch_stride, k_int8_head_stride, k_int8_seq_stride, k_int8_k_stride,
    k_scale_batch_stride, k_scale_head_stride,
    BLKK: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    block_id = pid % num_blocks

    batch_id = block_id // (k_head_stride // k_k_stride)
    head_id = (block_id % (k_head_stride // k_k_stride)) // (k_seq_stride // k_k_stride)
    seq_id = block_id % (k_seq_stride // k_k_stride)

    k_offset = batch_id * k_batch_stride + head_id * k_head_stride + seq_id * k_seq_stride
    k_int8_offset = batch_id * k_int8_batch_stride + head_id * k_int8_head_stride + seq_id * k_int8_seq_stride
    k_scale_offset = batch_id * k_scale_batch_stride + head_id * k_scale_head_stride

    k_block = tl.load(k_ptr + k_offset + tl.arange(0, BLKK), mask=tl.arange(0, BLKK) < BLKK, other=0.0)
    max_abs = tl.max(tl.abs(k_block), axis=0)
    scale = 127.0 / max_abs
    k_int8_block = tl.round(k_block * scale)
    k_int8_block = tl.to_int8(k_int8_block)

    tl.store(k_int8_ptr + k_int8_offset + tl.arange(0, BLKK), k_int8_block, mask=tl.arange(0, BLKK) < BLKK)
    tl.store(k_scale_ptr + k_scale_offset, scale)

import torch

def per_block_int8(q, k, BLKQ, BLKK):
    # Reshape the input tensors to handle 3D or 4D tensors uniformly
    q_shape = q.shape
    k_shape = k.shape
    if len(q_shape) == 3:
        q = q.unsqueeze(1)
    if len(k_shape) == 3:
        k = k.unsqueeze(1)

    batch_size, num_heads, seq_len, qk_dim = q.shape

    # Initialize the output tensors
    q_int8 = torch.empty((batch_size, num_heads, seq_len, qk_dim), dtype=torch.int8, device=q.device)
    k_int8 = torch.empty((batch_size, num_heads, seq_len, qk_dim), dtype=torch.int8, device=k.device)
    q_scale = torch.empty((batch_size, num_heads, seq_len), dtype=torch.float32, device=q.device)
    k_scale = torch.empty((batch_size, num_heads, seq_len), dtype=torch.float32, device=k.device)

    # Calculate the grid configuration
    grid = (batch_size * num_heads * seq_len,)

    # Launch the Triton kernels
    q_kernel_per_block_int8[grid](
        q, q_int8, q_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(2), q_int8.stride(3),
        q_scale.stride(0), q_scale.stride(1),
        BLKQ
    )

    k_kernel_per_block_int8[grid](
        k, k_int8, k_scale,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
        k_scale.stride(0), k_scale.stride(1),
        BLKK
    )

    # Squeeze the dimensions if the input was 3D
    if len(q_shape) == 3:
        q_int8 = q_int8.squeeze(1)
        q_scale = q_scale.squeeze(1)
    if len(k_shape) == 3:
        k_int8 = k_int8.squeeze(1)
        k_scale = k_scale.squeeze(1)

    return q_int8, k_int8, q_scale, k_scale
