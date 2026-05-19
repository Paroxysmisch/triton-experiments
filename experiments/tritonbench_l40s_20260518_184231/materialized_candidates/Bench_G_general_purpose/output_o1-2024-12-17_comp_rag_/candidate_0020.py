import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 16}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 64}, num_warps=8),
        triton.Config({'BT': 128}, num_warps=8),
    ],
    key=['T']
)
@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    s_ptr,  # [B*H, T]
    o_ptr,  # [B*H, T]
    stride_t,
    T: tl.constexpr,
    BT: tl.constexpr
):
    # Identify which (B, H) pair this program is responsible for
    i_bh = tl.program_id(0)

    # Prepare a small matrix to compute prefix sums in a block
    idx = tl.arange(0, BT)
    m_s = tl.where(idx[:, None] >= idx[None, :], 1.0, 0.0).to(tl.float32)

    # Initialize the accumulator keeping track of sums of subsequent blocks
    partial_sum = tl.float32(0.0)

    # Loop over T in blocks of size BT, from the end to the beginning
    num_blocks = (T + BT - 1) // BT
    for block_idx in range(num_blocks - 1, -1, -1):
        offset = block_idx * BT

        # Create block pointers for input s and output o
        p_s = tl.make_block_ptr(
            base=s_ptr + i_bh * BT * stride_t,  # offset by B*H slice
            shape=(T,),
            strides=(stride_t,),
            offsets=(offset,),
            block_shape=(BT,),
            order=(0,)
        )
        p_o = tl.make_block
