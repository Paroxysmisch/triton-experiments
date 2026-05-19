import triton
import triton.language as tl
import torch

@triton.jit
def chunk_global_reversed_cumsum_vector_kernel(s_ptr, z_ptr, m_s_ptr, B, H, T, S, BT, BS, **meta):
    # Calculate block indices
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    s_idx = tl.program_id(2)
    
    # Calculate start indices for each dimension
    batch_start = b_idx * BT
    head_start = h_idx
    spatial_start = s_idx * BS
    
    # Initialize block-wise cumulative sum to zero
    b_z = tl.zeros([BT, BS], dtype=tl.float32)
    
    # Iterate over time dimension in reverse
    for t_offset in range(0, T, BT):
        t_idx = T - t_offset - BT - 1  # Reverse index
        
        # Load input block
        b_s = tl.load(s_ptr + (batch_start * H * T * S) + (head_start * T * S) + (t_idx * S) + spatial_start)
        
        # Load mask block
        m_s = tl.load(m_s_ptr + (batch_start * H * T * S) + (head_start * T * S) + (t_idx * S) + spatial_start)
        
        # Update block-wise cumulative sum using mask and input
        b_z += tl.dot(m_s, b_s)
        
        # Store result in output tensor
        tl.store(z_ptr + (batch_start * H * T * S) + (head_start * T * S) + (t_idx * S) + spatial_start, b_z)

def chunk_global_reversed_cumsum_vector(s, m_s, BT, BS):
    B, H, T, S = s.shape
    z = torch.zeros_like(s)
    
    # Launch the Triton kernel
    grid = (B, H, S // BS)
    chunk_global_reversed_cumsum_vector_kernel[grid](
        s, z, m_s,
        B, H, T, S,
        BT, BS
    )
    
    return z

# Example usage
B, H, T, S = 2, 4, 8, 16
BT, BS = 2, 4
s = torch.randn(B, H, T, S, device='cuda')
m_s = torch.ones(B, H, T, S, device='cuda')  # Example mask
z = chunk_global_reversed_cumsum_vector(s, m_s, BT, BS)
print(z)
