import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BT': 1, 'BS': 64}, num_warps=1),
        triton.Config({'BT': 1, 'BS': 128}, num_warps=2),
        triton.Config({'BT': 2, 'BS': 64}, num_warps=2),
        triton.Config({'BT': 2, 'BS': 128}, num_warps=4),
        triton.Config({'BT': 4, 'BS': 64}, num_warps=4),
        triton.Config({'BT': 4, 'BS': 128}, num_warps=4),
    ],
    key=['Size']
)
@triton.jit
def chunk_global_cumsum_vector_kernel(
    s_ptr, z_ptr,
    Batch, Head, Time, Size,
    stride_batch, stride_head, stride_time, stride_size,
    BT: tl.constexpr, BS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    # Calculate current block's start indices
    block_start_time = pid_t * BT
    block_start_size = 0  # Process entire Size dimension per block (BS >= Size)

    # Create block pointer for input tensor s
    s_block_ptr = tl.make_block_ptr(
        base=s_ptr + pid_b * stride_batch + pid_h * stride_head,
        shape=(Time, Size),
        strides=(stride_time, stride_size),
        offsets=(block_start_time, block_start_size),
        block_shape=(BT, BS),
        order=(1, 0),
    )
    b_s = tl.load(s_block_ptr, boundary_check=(0, 1)).to(tl.float32)

    # Create lower triangular mask for cumulative sum within the block
    cols = tl.arange(0, BS)
    rows = cols[:, None]
    m_s = tl.where(rows >= cols, 1.0, 0.0)

    # Compute block-level cumulative sum via matrix multiplication
    b_cumsum = tl.dot(b_s, m_s)

    # Create block pointer for output tensor z and store result
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr + pid_b * stride_batch + pid_h * stride_head,
        shape=(Time, Size),
        strides=(stride_time, stride_size),
        offsets=(block_start_time, block_start_size),
        block_shape=(BT, BS),
        order=(1, 0),
    )
    tl.store(z_block_ptr, b_cumsum)

def chunk_global_cumsum_vector(s: torch.Tensor, BT: int = 64, BS: int = 64) -> torch.Tensor:
    Batch, Head, Time, Size = s.shape
    z = torch.empty_like(s)
    grid = (Batch, Head, triton.cdiv(Time, BT))
    chunk_global_cumsum_vector_kernel[grid](
        s, z,
        Batch, Head, Time, Size,
        s.stride(0), s.stride(1), s.stride(2), s.stride(3),
        BT, BS
    )
    return z
