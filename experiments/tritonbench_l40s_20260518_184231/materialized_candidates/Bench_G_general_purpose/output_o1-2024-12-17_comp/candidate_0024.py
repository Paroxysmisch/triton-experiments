import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    s_ptr, o_ptr,
    B, H, T,
    stride_sB, stride_sH, stride_sT,
    stride_oB, stride_oH, stride_oT,
    BLOCK_SIZE: tl.constexpr
):
    # Each block handles one (b, h) pair
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H

    # Accumulator for sums beyond the current block
    b_z = tl.float32(0)
    offset = T

    # Process T dimension in chunks from the end to the start
    while offset > 0:
        offset_start = tl.max(offset - BLOCK_SIZE, 0)
        chunk_size = offset - offset_start

        # Create index range for current chunk
        idx_range = tl.arange(0, BLOCK_SIZE)
        real_idx = offset_start + idx_range
        mask = idx_range < chunk_size

        # Load the current chunk from s
        values = tl.load(
            s_ptr + b * stride_sB + h * stride_sH + real_idx * stride_sT,
            mask=mask,
            other=0.0
        )

        # Compute reversed cumulative sum within the chunk
        # and add b_z (sum of subsequent chunks)
        local_csum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        rev_acc = tl.float32(0)
        for j in range(BLOCK_SIZE):
            reverse_idx = BLOCK_SIZE - 1 - j
            val = tl.where(reverse_idx < chunk_size, values[reverse_idx], 0.0)
            rev_acc = rev_acc + val
            # Write partial reversed cumsum + b_z into local storage
            cval = b_z + rev_acc
            local_csum = tl.where(
                reverse_idx == idx_range,
                cval,
                local_csum
            )

        # Store the results into o
        tl.store(
            o_ptr + b * stride_oB + h * stride_oH + real_idx * stride_oT,
            local_csum,
            mask=mask
        )

        # Update b_z by adding the sum of this chunk
        b_z += rev_acc
        offset = offset_start

def chunk_global_reversed_cumsum_scalar(s):
    # s is a torch.Tensor of shape (B, H, T)
    B, H, T = s.shape
    o = s.new_empty(s.shape
