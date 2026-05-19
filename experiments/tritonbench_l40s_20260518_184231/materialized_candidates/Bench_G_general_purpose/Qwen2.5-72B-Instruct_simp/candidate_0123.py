import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_vector_kernel(
    s_ptr,  # pointer to the input tensor
    z_ptr,  # pointer to the output tensor
    m_s_ptr,  # pointer to the mask matrix
    B, H, T, S,  # dimensions of the input tensor
    BT, BS,  # block sizes for temporal and spatial dimensions
    stride_s_B, stride_s_H, stride_s_T, stride_s_S,  # strides for the input tensor
    stride_z_B, stride_z_H, stride_z_T, stride_z_S,  # strides for the output tensor
    stride_m_T, stride_m_S,  # strides for the mask matrix
    BLOCK_SIZE_T: tl.constexpr,  # block size for temporal dimension
    BLOCK_SIZE_S: tl.constexpr,  # block size for spatial dimension
):
    # Get the program ID in the grid
    pid_B = tl.program_id(0)
    pid_H = tl.program_id(1)
    pid_T = tl.program_id(2)
    pid_S = tl.program_id(3)

    # Compute the block indices
    block_start_T = pid_T * BLOCK_SIZE_T
    block_start_S = pid_S * BLOCK_SIZE_S

    # Initialize the block-wise cumulative sum
    b_z = tl.zeros((BLOCK_SIZE_S,), dtype=tl.float32)

    # Iterate backward through time blocks
    for t in range(T - 1, block_start_T - 1, -BLOCK_SIZE_T):
        # Load the input block
        b_s = tl.load(s_ptr + pid_B * stride_s_B + pid_H * stride_s_H + t * stride_s_T + block_start_S * stride_s_S, 
                      mask=block_start_S + tl.arange(0, BLOCK_SIZE_S) < S, 
                      other=0.0)

        # Load the mask block
        b_m = tl.load(m_s_ptr + t * stride_m_T + block_start_S * stride_m_S, 
                      mask=block_start_S + tl.arange(0, BLOCK_SIZE_S) < S, 
                      other=0.0)

        # Compute the block-wise cumulative sum
        b_z = b_z + tl.dot(b_m, b_s)

        # Store the result in the output tensor
        tl.store(z_ptr + pid_B * stride_z_B + pid_H * stride_z_H + t * stride_z_T + block_start_S * stride_z_S, 
                 b_z, 
                 mask=block_start_S + tl.arange(0, BLOCK_SIZE_S) < S)

import torch

def chunk_global_reversed_cumsum_vector(s, m_s, BT, BS):
    B, H, T, S = s.shape
    z = torch.zeros_like(s)

    # Define the grid and block sizes
    grid = (B, H, (T + BT - 1) // BT, (S + BS - 1) // BS)
    block = (BS,)

    # Launch the kernel
    chunk_global_reversed_cumsum_vector_kernel[grid, block](
        s, z, m_s, B, H, T, S, BT, BS,
        s.stride(0), s.stride(1), s.stride(2), s.stride(3),
        z.stride(0), z.stride(1), z.stride(2), z.stride(3),
        m_s.stride(0), m_s.stride(1),
        BT, BS
    )

    return z
