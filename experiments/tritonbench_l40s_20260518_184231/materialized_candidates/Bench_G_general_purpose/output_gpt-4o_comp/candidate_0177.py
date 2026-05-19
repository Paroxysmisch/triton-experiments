import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BT': 32, 'BS': 32}, num_warps=4),
        triton.Config({'BT': 64, 'BS': 64}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def chunk_global_cumsum_vector_kernel(
    s_ptr, z_ptr,
    B, H, T, S,
    stride_bs, stride_bt, stride_bh, stride_bb,
    stride_zs, stride_zt, stride_zh, stride_zb,
    BT: tl.constexpr, BS: tl.constexpr
):
    # Calculate the program's position
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    block_time_id = tl.program_id(2)

    # Calculate the offset for the current block
    block_time_offset = block_time_id * BT

    # Create a mask for the lower triangular matrix
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    # Pointer to the start of the block in `s`
    s_block_ptr = tl.make_block_ptr(
        s_ptr + batch_id * stride_bb + head_id * stride_bh + block_time_offset * stride_bt,
        (BT, S),
        (stride_bt, stride_bs),
        (B, H, T, S)
    )

    # Load the block from `s` and cast to float32
    b_s = tl.load(s_block_ptr).to(tl.float32)

    # Compute the block-level cumulative sum
    b_c = tl.dot(m_s, b_s)

    # Pointer to the start of the block in `z`
    z_block_ptr = tl.make_block_ptr(
        z_ptr + batch_id * stride_zb + head_id * stride_zh + block_time_offset * stride_zt,
        (BT, S),
        (stride_zt, stride_zs),
        (B, H, T, S)
    )

    # Store the result back to `z`
    tl.store(z_block_ptr, b_c)

def chunk_global_cumsum_vector(s: torch.Tensor) -> torch.Tensor:
    # Check input dimensions
    assert s.ndim == 4, "Input tensor must be 4D [Batch, Head, Time, Size]"
    B, H, T, S = s.shape

    # Output tensor
    z = torch.empty_like(s)

    # Define grid dimensions
    grid = (B, H, (T + 31) // 32)  # Assuming BT = 32

    # Launch the kernel
    chunk_global_cumsum_vector_kernel[grid](
        s, z,
        B, H, T, S,
        s.stride(3), s.stride(2), s.stride(1), s.stride(0),
        z.stride(3), z.stride(2), z.stride(1), z.stride(0),
        BT=32, BS=32  # Example block sizes, can be tuned
    )

    return z
