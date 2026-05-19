import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 8}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 16}),
    ],
    key=['S'],
)
@triton.jit
def chunk_global_cumsum_vector_kernel(
    z_ptr, s_ptr,                                # pointers to output and input tensors
    stride_z_b, stride_z_h, stride_z_t,          # strides for output tensor
    stride_s_b, stride_s_h, stride_s_t,          # strides for input tensor
    B, H, T, S,                                  # tensor dimensions
    BT: tl.constexpr, BS: tl.constexpr,         # block sizes for time and feature dimensions
    BLOCK_SIZE: tl.constexpr,                    # size of parallel processing block
    NUM_WARPS: tl.constexpr,                     # number of warps for parallel execution
):
    # Get program ID for batch, head, and time dimensions
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    # Create lower triangular mask for cumsum operation
    offs_m = tl.arange(0, BS)
    offs_n = tl.arange(0, BS)
    m_s = tl.where(offs_m[:, None] >= offs_n[None, :], 1.0, 0.0)

    # Initialize running sum
    b_z = tl.zeros([BS], dtype=tl.float32)

    # Calculate base pointers for current batch and head
    base_s_ptr = s_ptr + pid_b * stride_s_b + pid_h * stride_s_h
    base_z_ptr = z_ptr + pid_b * stride_z_b + pid_h * stride_z_h

    # Process blocks in time dimension
    for t in range(pid_t * BT, min((pid_t + 1) * BT, T), BS):
        # Create block pointers for current time slice
        s_block_ptr = tl.make_block_ptr(
            base_s_ptr + t * stride_s_t,
            shape=(BS, S),
            strides=(1, stride_s_t),
            offsets=(0, 0),
            block_shape=(BS, BS),
            order=(1, 0)
        )

        # Load input block
        b_s = tl.load(s_block_ptr, boundary_check=(0, 1))
        b_s = b_s.to(tl.float32)

        # Compute cumulative sum for current block
        b_c = tl.dot(m_s, b_s)
        b_c = b_c + b_z[:, None]

        # Store result
        z_block_ptr = tl.make_block_ptr(
            base_z_ptr + t * stride_z_t,
            shape=(BS, S),
            strides=(1, stride_z_t),
            offsets=(0, 0),
            block_shape=(BS, BS),
            order=(1, 0)
        )
        tl.store(z_block_ptr, b_c, boundary_check=(0, 1))

        # Update running sum
        b_z = tl.sum(b_s, axis=1)

def chunk_global_cumsum_vector(s: torch.Tensor) -> torch.Tensor:
    """
    Compute global cumulative sum over blocks in a 4D tensor.
    
    Args:
        s: Input tensor of shape [Batch, Head, Time, Size]
    
    Returns:
        z: Output tensor of same shape with cumulative sum applied
    """
    assert len(s.shape) == 4, "Input tensor must be 4D [Batch, Head, Time, Size]"
    B, H, T, S = s.shape
    
    # Create output tensor
    z = torch.empty_like(s)
    
    # Define block sizes
    BT = 8  # Block size for time dimension
    BS = min(128, S)  # Block size for feature dimension
    
    # Calculate grid dimensions
    grid = (B, H, triton.cdiv(T, BT))
    
    # Get tensor strides
    stride_z_b, stride_z_h, stride_z_t, stride_z_s = z.stride()
    stride_s_b, stride_s_h, stride_s_t, stride_s_s = s.stride()
    
    # Launch kernel
    chunk_global_cumsum_vector_kernel[grid](
        z_ptr=z.data_ptr(),
        s_ptr=s.data_ptr(),
        stride_z_b=stride_z_b,
        stride_z_h=stride_z_h,
        stride_z_t=stride_z_t,
        stride_s_b=stride_s_b,
        stride_s_h=stride_s_h,
        stride_s_t=stride_s_t,
        B=B, H=H, T=T, S=S,
        BT=BT,
        BS=BS,
    )
    
    return z
