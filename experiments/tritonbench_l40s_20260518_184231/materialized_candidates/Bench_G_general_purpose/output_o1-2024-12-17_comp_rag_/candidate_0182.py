import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_T": 32, "BLOCK_SIZE_S": 32, "num_warps": 1}),
        triton.Config({"BLOCK_SIZE_T": 64, "BLOCK_SIZE_S": 32, "num_warps": 2}),
        triton.Config({"BLOCK_SIZE_T": 128, "BLOCK_SIZE_S": 64, "num_warps": 4}),
    ],
    key=["T", "S"],
)
@triton.jit
def chunk_global_cumsum_vector_kernel(
    s_ptr, z_ptr,
    B, H, T, S,
    stride_sB, stride_sH, stride_sT, stride_sS,
    stride_zB, stride_zH, stride_zT, stride_zS,
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    b_offset = pid_b * stride_sB
    h_offset = pid_h * stride_sH
    t_block_start = pid_t * BLOCK_SIZE_T
    # Pointers to input/output blocks
    s_block_ptr = s_ptr + b_offset + h_offset + t_block_start * stride_sT
    z_block_ptr = z_ptr + (pid_b * stride_zB) + (pid_h * stride_zH) + (t_block_start * stride_zT)

    # Offsets in time and size
    tt = tl.arange(0, BLOCK_SIZE_T)
    ss = tl.arange(0, BLOCK_SIZE_S)
    # Masks for valid loads
    mask_t = tt + t_block_start < T
    mask_s = ss < S

    # 2D expansion
    t_mask = mask_t[:, None]
    s_mask = mask_s[None, :]

    # Build a lower-triangular mask for cumsum in a block of size BLOCK_SIZE_T
    tri_t = tl.arange(0, BLOCK_SIZE_T)[:, None]
    tri_s = tl.arange(0, BLOCK_SIZE_T)[None, :]
    lower_mask = tri_t >= tri_s

    # Load data from s
    s_values = tl.load(
        tl.make_block_ptr(
            base=s_block_ptr,
            shape=(T, S),
            strides=(stride_sT, stride_sS),
            offsets=(tt[:, None], ss[None, :]),
            block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_S),
            order=(0, 1),
        ),
        mask=t_mask[:, None] & s_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # We will compute local cumsum in time dimension
    # Split the cumsum into partial sums using matrix multiply with the lower-triangular mask
    # Build the NxN matrix for the time dimension
    s_values_t = s_values  # [BLOCK_SIZE_T, BLOCK_SIZE_S]
    # Expand dimension for matmul: [BLOCK_SIZE_T, BLOCK_SIZE_S] -> [BLOCK_SIZE_T, BLOCK_SIZE_T], [BLOCK_SIZE_T, BLOCK_SIZE_S]
    # We do cumsum for each s index separately
    # We'll replicate each column of s_values_t across the second matrix dimension
    # to multiply with the lower-triangular matrix
    # "time_mat" is a one-hot version of s_values_t repeated across diagonal to do "cumsum"
    # For performance in real code, one might use a prefix-sum algorithm. Here we do dot for demonstration.

    # Construct the block-time lower triangle as float32
    tri_mat = tl.where(lower_mask, 1.0, 0.0)  # [BLOCK_SIZE_T, BLOCK_SIZE_T]
    csum_out = tl.zeros([BLOCK_SIZE_T, BLOCK_SIZE_S], dtype=tl.float32)

    # For each column in s_values_t (per size index), we can do:
    # csum_out[:, col] = tri_mat @ s_values_t[:, col]
    # We replicate columns of s_values_t to multiply. We do a loop over columns due to Triton matmul restrictions.

    for col in range(BLOCK_SIZE_S):
        col_vector = s_values_t[:, col]  # [BLOCK_SIZE_T]
        col_vector_2d = tl.broadcast_to(col_vector[:, None], [BLOCK_SIZE_T, BLOCK_SIZE_T])
        partial_sum = tl.dot(tri_mat, col_vector_2d)
        csum_out[:, col] = partial_sum[:, 0]

    # In a global cumsum, we need to incorporate the sum from all previous blocks in time
    # We'll read the last partial sum from the previous block if pid_t > 0
    # Accumulate that offset
    if pid_t > 0:
        prev_block_end_ptr = z_ptr + (pid_b * stride_zB) + (pid_h * stride_zH) + ((t_block_start - 1) * stride_zT)
        # We'll load the last row from the previous block
        # index last row in [BLOCK_SIZE_T, S], i.e. offset = BLOCK_SIZE_T-1
        prev_tt_idx = BLOCK_SIZE_T - 1
        prev_mask_t = (prev_tt_idx + (t_block_start - 1) * BLOCK_SIZE_T) < T
        prev_s_values = tl.load(
            tl.make_block_ptr(
                base=prev_block_end_ptr,
                shape=(T, S),
                strides=(stride_zT, stride_zS),
                offsets=(prev_tt_idx, ss),
                block_shape=(1, BLOCK_SIZE_S),
                order=(0, 1),
            ),
            mask=prev_mask_t & s_mask,
            other=0.0,
        ).to(tl.float32)
        # This is the final cumsum from the previous block
        offset = prev_s_values
        # Add offset to the entire csum_out
        csum_out = csum_out + offset[None, :]

    # Store results back to z
    tl.store(
        tl.make_block_ptr(
            base=z_block_ptr,
            shape=(T, S),
            strides=(stride_zT, stride_zS),
            offsets=(tt[:, None], ss[None, :]),
            block_shape=(BLOCK_SIZE_T, BLOCK_SIZE_S),
            order=(0, 1),
        ),
        csum_out,
        mask=t_mask[:, None] & s_mask[None, :],
    )


def chunk_global_cumsum_vector(s: torch.Tensor) -> torch.Tensor:
    assert s.ndim == 4, "Input tensor must have shape [B, H, T, S]"
    B, H, T, S_ = s.shape
    z = torch.empty_like(s, dtype=torch.float32)

    grid = lambda META: (
        B,
        H,
        (T + META["BLOCK_SIZE_T"] - 1) // META["BLOCK_SIZE_T"],
    )

    stride_sB = s.stride(0)
    stride_sH = s.stride(1)
    stride_sT = s.stride(2)
    stride_sS = s.stride(3)

    stride_zB = z.stride(0)
    stride_zH = z.stride(1)
    stride_zT = z.stride(2)
    stride_zS = z.stride(3)

    with torch.cuda.device(s.device.index):
        chunk_global_cumsum_vector_kernel[grid](
            s, z,
            B, H, T, S_,
            stride_sB, stride_sH, stride_sT, stride_sS,
            stride_zB, stride_zH, stride_zT, stride_zS,
        )
    return z
