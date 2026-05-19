import triton
import triton.language as tl

# Triton kernel for reversed cumulative sum
@triton.jit
def chunk_global_reversed_cumsum_vector_kernel(
    s_ptr,  # Pointer to the input tensor
    z_ptr,  # Pointer to the output tensor
    B, H, T, S,  # Dimensions of the input tensor
    BT,  # Block size for time dimension
    BS,  # Block size for spatial dimension
    dtype: tl.dtype,  # Data type of the tensors
    BLOCK_M: tl.constexpr,  # Block size for the time dimension
    BLOCK_N: tl.constexpr,  # Block size for the spatial dimension
    BLOCK_K: tl.constexpr,  # Block size for the feature dimension
):
    # Get the current block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = tl.program_id(3)

    # Compute the block start indices
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N
    rb = pid_b * B
    rh = pid_h * H

    # Initialize the cumulative sum block
    b_z = tl.zeros((BLOCK_M, BLOCK_N), dtype=dtype)

    # Iterate backwards over time blocks
    for t_block in range(tl.cdiv(T, BT) - 1, -1, -1):
        rt = t_block * BT

        # Create block pointers for input and output
        s_block_ptr = s_ptr + (rb + pid_b) * S + (rh + pid_h) * T * S + rt * S + rn
        z_block_ptr = z_ptr + (rb + pid_b) * S + (rh + pid_h) * T * S + rt * S + rn

        # Load the input block
        b_s = tl.load(s_block_ptr, mask=rm + tl.arange(0, BLOCK_M) < T, other=0.0)

        # Compute the masked dot product with a lower triangular mask
        m_s = tl.full((BLOCK_M, BLOCK_N), 1.0, dtype=dtype)
        m_s = tl.where(tl.arange(0, BLOCK_M)[:, None] >= tl.arange(0, BLOCK_N), m_s, 0.0)
        b_s = b_s * m_s

        # Update the cumulative sum block
        b_z = b_z + tl.sum(b_s, axis=1)

        # Store the result in the output block
        tl.store(z_block_ptr, b_z, mask=rm + tl.arange(0, BLOCK_M) < T)

# Wrapper function to call the Triton kernel
def chunk_global_reversed_cumsum_vector(s, dtype):
    B, H, T, S = s.shape
    BS = 32  # Spatial block size

    # Initialize the output tensor
    z = tl.zeros_like(s, dtype=dtype)

    # Configure the kernel launch
    grid = (tl.cdiv(T, BS), tl.cdiv(S, BS), B, H)
    block = (BS, BS, 1)

    # Launch the kernel
    chunk_global_reversed_cumsum_vector_kernel[grid, block](
        s, z, B, H, T, S, BS, dtype, BS, BS, S
    )

    return z
