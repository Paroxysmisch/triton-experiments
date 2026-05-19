import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BT": 16, "BS": 16}, num_warps=1),
        triton.Config({"BT": 16, "BS": 32}, num_warps=2),
        triton.Config({"BT": 32, "BS": 32}, num_warps=4),
        triton.Config({"BT": 32, "BS": 64}, num_warps=8),
    ],
    key=["S"],
)
@triton.jit
def chunk_global_cumsum_vector_kernel(
    s_ptr, z_ptr,
    B, H, T, S,         # Tensor shape
    stride_bs, stride_hs, stride_ts, stride_ss,   # Input strides
    stride_bz, stride_hz, stride_tz, stride_sz,   # Output strides
    BT: tl.constexpr,   # Block size in time dimension
    BS: tl.constexpr,   # Block size in size dimension
):
    # Program IDs to index batch and head
    b_id = tl.program_id(axis=0)
    h_id = tl.program_id(axis=1)

    # Create a lower-triangular mask for block-level cumsum
    rows = tl.arange(0, BS)
    cols = tl.arange(0, BS)
    mask = rows[:, None] >= cols[None, :]

    # Offsets for batch/head in input and output
    s_offset = b_id * stride_bs + h_id * stride_hs
    z_offset = b_id * stride_bz + h_id * stride_hz

    # Running sum buffer across blocks
    running_sum = tl.zeros([BS], dtype=tl.float32)

    # Process the time dimension in chunks of BT
    t_block = 0
    while t_block < T:
        # Compute actual block size (handle remainder)
        current_bt = tl.minimum(BT, T - t_block)

        # For each row in [0..current_bt), we load a [BS]-wide slice
        # Create block pointers for each row
        row_idx_t = t_block + tl.arange(0, current_bt)
        ptr_s_block = tl.make_block_ptr(
            base=s_ptr + s_offset,
            shape=(T, S),
            strides=(stride_ts, stride_ss),
            offsets=(row_idx_t, 0),
            block_shape=(current_bt, BS),
            order=(0, 1),
        )
        ptr_z_block = tl.make_block_ptr(
            base=z_ptr + z_offset,
            shape=(T, S),
            strides=(stride_tz, stride_sz),
            offsets=(row_idx_t, 0),
            block_shape=(current_bt, BS),
            order=(0, 1),
        )

        # Load the slice for the entire block
        b_s = tl.load(ptr_s_block, mask=(rows[None, :] < current_bt) & (cols[None, :] < BS))
        b_s = b_s.to(tl.float32)

        # Apply block-level cumulative sum (matrix mul with lower-tri mask)
        # shape(b_s): current_bt x BS, shape(mask): BS x BS (we broadcast to multiply)
        # We'll transpose the block so that the cumsum is done across columns
        b_s_t = tl.trans(b_s)
        b_c = tl.dot(mask.to(b_s_t.dtype), b_s_t)
        b_c_t = tl.trans(b_c)

        # Add running_sum to each row in this block for the next chunk's continuity
        # broadcast across each row -> shape of running_sum: [BS]
        b_c_t = b_c_t + running_sum[None, :]

        # Store result
        tl.store(ptr_z_block, b_c_t, mask=(rows[None, :] < current_bt) & (cols[None, :] < BS))

        # Update running_sum with the last row's sums of this block
        last_row_c = tl.load(ptr_z_block, mask=(rows < 1) & (cols[None, :] < BS), boundary_check=[False, True])
        # last_row_c is shape (1, BS); after load transposing or indexing we just take it as running_sum
        running_sum = last_row_c[0, :]

        # Move to next time block
        t_block += current_bt


def chunk_global_cumsum_vector(s: torch.Tensor):
    """
    4D input tensor s: [Batch, Head, Time, Size]
    Returns a tensor z with the same shape
    """
    B, H, T, S = s.shape
    # Create output
    z = torch.empty_like(s)
    # Launch kernel with a 2D grid over (B, H)
    grid = (B, H)
    chunk_global_cumsum_vector_kernel[grid](
        s, z,
        B, H, T, S,
        s.stride(0), s.stride(1), s.stride(2), s.stride(3),
        z.stride(0), z.stride(1), z.stride(2), z.stride(3),
    )
    return z
