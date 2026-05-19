import triton
import triton.language as tl
import torch

# Define the kernel
@triton.jit
def chunk_global_reversed_cumsum_vector_kernel(
    s_ptr, z_ptr, B, H, T, S, BT, BS,
    BLOCK_SIZE: tl.constexpr,  # Spatial block size
):
    # Calculate block indices
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    t_block_idx = tl.program_id(2)

    # Compute the starting index for the current block
    t_start = (tl.cdiv(T, BT) - 1 - t_block_idx) * BT

    # Pointers for the input and output blocks
    s_offset = b_idx * H * T * S + h_idx * T * S + t_start * S
    z_offset = b_idx * H * T * S + h_idx * T * S + t_start * S

    # Initialize cumulative sum
    b_z = tl.zeros([BT, BS], dtype=tl.float32)

    # Loop over time blocks
    for t in range(BT):
        # Compute the current time index
        t_idx = t_start + t

        # Boundary check for time dimension
        if t_idx < T:
            # Load input block
            b_s = tl.load(s_ptr + s_offset + t * S, mask=True)

            # Create a lower triangular mask for reversed cumsum
            m_s = tl.arange(0, BS) >= (BS - 1 - t)

            # Compute masked dot product
            b_z += tl.where(m_s, b_s, tl.zeros_like(b_s))

            # Store the result in the output tensor
            tl.store(z_ptr + z_offset + t * S, b_z, mask=True)

# Wrapper function
def chunk_global_reversed_cumsum_vector(s: torch.Tensor, dtype=torch.float32):
    B, H, T, S = s.shape
    BS = 32  # Spatial block size
    BT = 32  # Time block size

    # Allocate output tensor
    z = torch.empty((B, H, T, S), dtype=dtype, device=s.device)

    # Launch the kernel
    grid = (B, H, tl.cdiv(T, BT))
    num_warps = 4  # Tune this parameter for your hardware

    chunk_global_reversed_cumsum_vector_kernel[grid](
        s, z, B, H, T, S, BT, BS,
        num_warps=num_warps,
        BLOCK_SIZE=BS
    )

    return z

# Example usage
s = torch.randn(2, 3, 128, 64, device='cuda', dtype=torch.float32)
z = chunk_global_reversed_cumsum_vector(s)
print(z)
