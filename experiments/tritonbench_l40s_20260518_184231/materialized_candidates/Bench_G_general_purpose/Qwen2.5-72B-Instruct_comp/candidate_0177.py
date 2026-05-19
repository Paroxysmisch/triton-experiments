import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BT': 128, 'BS': 32}, num_warps=4),
        triton.Config({'BT': 256, 'BS': 32}, num_warps=8),
        triton.Config({'BT': 512, 'BS': 32}, num_warps=16),
    ],
    key=['S']
)
@triton.jit
def chunk_global_cumsum_vector_kernel(
    s_ptr,  # Pointer to the input tensor
    z_ptr,  # Pointer to the output tensor
    B, H, T, S,  # Tensor dimensions
    BT, BS,  # Block time and block size
    stride_s_B, stride_s_H, stride_s_T, stride_s_S,  # Strides for input tensor
    stride_z_B, stride_z_H, stride_z_T, stride_z_S,  # Strides for output tensor
    BLOCK_TIME: tl.constexpr,  # Block time
    BLOCK_SIZE: tl.constexpr  # Block size
):
    # Determine the program's position
    pid_b = tl.program_id(0)  # Batch
    pid_h = tl.program_id(1)  # Head
    pid_t = tl.program_id(2)  # Time

    # Check if the program ID is within bounds
    if pid_b >= B or pid_h >= H or pid_t >= T:
        return

    # Create a lower triangular mask
    m_s = tl.full((BLOCK_TIME, BLOCK_TIME), 0, tl.int32)
    for i in range(BLOCK_TIME):
        for j in range(i + 1):
            m_s[i, j] = 1

    # Initialize the running sum
    b_z = tl.zeros((BLOCK_TIME, BLOCK_SIZE), dtype=tl.float32)

    # Process each block in the time dimension
    for t in range(pid_t * BLOCK_TIME, (pid_t + 1) * BLOCK_TIME, BLOCK_TIME):
        # Create a pointer to the relevant data slice in s
        s_block_ptr = tl.make_block_ptr(
            base=s_ptr,
            shape=(B, H, T, S),
            strides=(stride_s_B, stride_s_H, stride_s_T, stride_s_S),
            offsets=(pid_b, pid_h, t, 0),
            block_shape=(1, 1, BLOCK_TIME, BLOCK_SIZE),
            order=(3, 2, 1, 0)
        )

        # Load data from s
        b_s = tl.load(s_block_ptr)

        # Convert to float32 for precision
        b_s = tl.to(b_s, tl.float32)

        # Compute the block-level cumulative sum
        b_c = tl.dot(m_s, b_s)

        # Add the running sum
        b_c += b_z

        # Create a pointer to the relevant data slice in z
        z_block_ptr = tl.make_block_ptr(
            base=z_ptr,
            shape=(B, H, T, S),
            strides=(stride_z_B, stride_z_H, stride_z_T, stride_z_S),
            offsets=(pid_b, pid_h, t, 0),
            block_shape=(1, 1, BLOCK_TIME, BLOCK_SIZE),
            order=(3, 2, 1, 0)
        )

        # Store the result back to the output tensor z
        tl.store(z_block_ptr, b_c)

        # Update the running sum
        b_z = b_c

import torch
import triton
import triton.language as tl

def chunk_global_cumsum_vector(s: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    assert s.is_cuda, "Input tensor must be on the GPU"

    # Get the tensor dimensions
    B, H, T, S = s.shape

    # Create the output tensor
    z = torch.empty_like(s)

    # Set the block size and number of warps
    BT = 128  # Block time
    BS = 32   # Block size

    # Set the grid dimensions
    grid = (B, H, (T + BT - 1) // BT)

    # Set the strides for the input and output tensors
    stride_s_B, stride_s_H, stride_s_T, stride_s_S = s.stride()
    stride_z_B, stride_z_H, stride_z_T, stride_z_S = z.stride()

    # Call the kernel
    chunk_global_cumsum_vector_kernel[grid](
        s, z, B, H, T, S, BT, BS,
        stride_s_B, stride_s_H, stride_s_T, stride_s_S,
        stride_z_B, stride_z_H, stride_z_T, stride_z_S,
        BLOCK_TIME=BT, BLOCK_SIZE=BS
    )

    return z
