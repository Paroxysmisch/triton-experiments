import triton
import triton.language as tl
import torch

@triton.jit
def chunk_global_cumsum_scalar_kernel(
    s_ptr,  # *F32[B*H*T]
    o_ptr,  # *F32[B*H*T]
    B,      # i32
    H,      # i32
    T,      # i32
    block_size: tl.constexpr
):
    pid = tl.program_id(0)
    b_idx = pid // H
    h_idx = pid % H

    # Base offset for this (b_idx, h_idx) in the flattened memory
    base_offset = (b_idx * H + h_idx) * T

    sum_carry = tl.zeros([1], dtype=tl.float32)
    block_start = 0

    # Loop over chunks of the T dimension
    while block_start < T:
        tid = tl.arange(0, block_size)
        offsets = base_offset + block_start + tid
        mask = tid + block_start < T

        # Load current chunk
        s_block_ptr = tl.make_block_ptr(
            base=s_ptr,
            shape=[T],
            strides=[1],
            offsets=offsets,
            block_shape=[block_size],
            order=[0]
        )
        data = tl.load(s_block_ptr, mask=mask, other=0.0)

        # Cumulative sum
        data = tl.cumsum(data, 0)
        data += sum_carry

        # Store result
        o_block_ptr = tl.make_block_ptr(
            base=o_ptr,
            shape=[T],
            strides=[1],
            offsets=offsets,
            block_shape=[block_size],
            order=[0]
        )
        tl.store(o_block_ptr, data, mask=mask)

        # Update carry
        valid_count = tl.sum(mask, axis=0)
        final_idx = valid_count - 1
        last_val_mask = tid == final_idx
        last_val = tl.where(last_val_mask, data, 0.0)
        sum_carry = tl.sum(last_val)

        block_start += block_size


def chunk_global_cumsum_scalar(s: torch.Tensor, dtype=None):
    if dtype is None:
        dtype = s.dtype
    B, H, T = s.shape
    z = torch.empty_like(s, dtype=dtype)
    grid = (B * H,)
    block_size = 128
    chunk_global_cumsum_scalar_kernel[grid](
        s, z, B, H, T,
        block_size=block_size,
        num_warps=4
    )
    return z
