import triton
import triton.language as tl
import torch

# Kernel function
@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(s_ptr, z_ptr, B, H, T, BT, stride_b, stride_h, stride_t):
    pid = tl.program_id(axis=0)
    # Calculate the batch and head indices
    b_idx = pid // H
    h_idx = pid % H

    # Offset pointers to the start of the batch and head
    s_offset = b_idx * stride_b + h_idx * stride_h
    z_offset = b_idx * stride_b + h_idx * stride_h

    # Create a pointer to the start of the last chunk
    t_start = T - BT

    # Initialize a cumulative sum variable
    cumsum = tl.zeros((BT,), dtype=tl.float32)

    # Iterate over chunks in reverse
    for t in range(t_start, -BT, -BT):
        # Load a chunk of data
        s_chunk = tl.load(s_ptr + s_offset + t * stride_t + tl.arange(0, BT))
        
        # Update the cumulative sum
        cumsum = cumsum + s_chunk
        
        # Store the cumulative sum
        tl.store(z_ptr + z_offset + t * stride_t + tl.arange(0, BT), cumsum)

# Wrapper function
def chunk_global_reversed_cumsum_scalar(s, BT):
    B, H, T = s.shape
    z = torch.empty_like(s)

    # Launch the kernel
    grid = (B * H,)
    chunk_global_reversed_cumsum_scalar_kernel[grid](
        s_ptr=s,
        z_ptr=z,
        B=B,
        H=H,
        T=T,
        BT=BT,
        stride_b=s.stride(0),
        stride_h=s.stride(1),
        stride_t=s.stride(2),
    )
    return z

# Example usage
B, H, T = 4, 3, 16  # Example dimensions
BT = 4  # Chunk size
s = torch.randn((B, H, T), device='cuda')  # Input tensor on GPU
z = chunk_global_reversed_cumsum_scalar(s, BT)
print(z)
