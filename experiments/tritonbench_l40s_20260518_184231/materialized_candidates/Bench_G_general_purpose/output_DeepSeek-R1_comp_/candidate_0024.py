import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    s_ptr,
    o_ptr,
    B,
    H,
    T,
    s_b_stride,
    s_h_stride,
    s_t_stride,
    o_b_stride,
    o_h_stride,
    o_t_stride,
    BT: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    s_ptr_bh = s_ptr + b * s_b_stride + h * s_h_stride
    o_ptr_bh = o_ptr + b * o_b_stride + h * o_h_stride

    num_blocks = (T + BT - 1) // BT
    b_z = 0.0

    for i in range(num_blocks):
        block_idx = num_blocks - 1 - i
        start_idx = block_idx * BT
        end_idx = tl.minimum(start_idx + BT, T)

        reverse_offsets = end_idx - 1 - tl.arange(0, BT)
        mask_rev_load = (reverse_offsets >= start_idx) & (reverse_offsets < T)
        s_block_rev = tl.load(
            s_ptr_bh + reverse_offsets * s_t_stride,
            mask=mask_rev_load,
            other=0.0,
        )

        # Compute prefix sum on reversed block
        prefix_sum = tl.associative_scan(s_block_rev, axis=0, combine_fn=tl.sum)

        # Reverse the prefix sum to get local reversed cumsum
        index = BT - 1 - tl.arange(0, BT)
        reversed_cumsum_block = prefix_sum[index]

        # Add accumulated sum and store
        o_block = reversed_cumsum_block + b_z
        write_offsets = start_idx + tl.arange(0, BT)
        mask_store = write_offsets < end_idx
        tl.store(
            o_ptr_bh + write_offsets * o_t_stride,
            o_block,
            mask=mask_store,
        )

        # Update accumulation variable
        sum_block = tl.sum(s_block_rev)
        b_z += sum_block

def chunk_global_reversed_cumsum_scalar(s: torch.Tensor):
    B, H, T = s.shape
    o = torch.empty_like(s)
    grid = (B * H,)

    s_b_stride = s.stride(0)
    s_h_stride = s.stride(1)
    s_t_stride = s.stride(2)
    o_b_stride = o.stride(0)
    o_h_stride = o.stride(1)
    o_t_stride = o.stride(2)

    BT = 128  # Tuning this value can optimize performance

    chunk_global_reversed_cumsum_scalar_kernel[grid](
        s, o, B, H, T,
        s_b_stride, s_h_stride, s_t_stride,
        o_b_stride, o_h_stride, o_t_stride,
        BT=BT,
    )
    return o
