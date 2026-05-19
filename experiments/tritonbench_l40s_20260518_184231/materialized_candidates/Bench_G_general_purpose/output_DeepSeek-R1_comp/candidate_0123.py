import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_vector_kernel(
    s_ptr, z_ptr,
    B, H, T, S,
    stride_s_b, stride_s_h, stride_s_t, stride_s_s,
    stride_z_b, stride_z_h, stride_z_t, stride_z_s,
    BT: tl.constexpr, BS: tl.constexpr
):
    # Program indices for batch, head, and spatial block
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    s_block_idx = tl.program_id(2)
    
    # Spatial block range
    s_start = s_block_idx * BS
    s_offsets = s_start + tl.arange(0, BS)
    s_mask = s_offsets < S
    
    # Initialize carry vector for each spatial position in the block
    carry = tl.zeros([BS], dtype=tl.float32)
    
    # Number of T blocks and loop over them in reverse
    num_T_blocks = tl.cdiv(T, BT)
    for t_block_idx in range(num_T_blocks - 1, -1, -1):
        t_start = t_block_idx * BT
        t_offsets = t_start + tl.arange(0, BT)
        t_mask = t_offsets < T
        
        # Create block pointers for input and output
        s_block_ptr = tl.make_block_ptr(
            base=s_ptr,
            shape=(B, H, T, S),
            strides=(stride_s_b, stride_s_h, stride_s_t, stride_s_s),
            offsets=(b_idx, h_idx, t_start, s_start),
            block_shape=(BT, BS),
            order=(1, 0)  # Adjusted for T and S dimensions
        )
        z_block_ptr = tl.make_block_ptr(
            base=z_ptr,
            shape=(B, H, T, S),
            strides=(stride_z_b, stride_z_h, stride_z_t, stride_z_s),
            offsets=(b_idx, h_idx, t_start, s_start),
            block_shape=(BT, BS),
            order=(1, 0)  # Adjusted for T and S dimensions
        )
        
        # Load input block, applying boundary checks and padding with zeros
        s_block = tl.load(s_block_ptr, boundary_check=(2, 3), padding_option="zero")
        
        # Compute reversed cumulative sum within the block
        reversed_block = tl.reverse(s_block, axis=0)
        scan = tl.associative_scan(reversed_block, axis=0, combine_fn=lambda a, b: a + b)
        reversed_cumsum = tl.reverse(scan, axis=0)
        
        # Add carry to each element in the block
        z_block = reversed_cumsum + carry
        
        # Store the result, converting to output dtype
        tl.store(z_block_ptr, z_block.to(z_ptr.dtype.element_ty), boundary_check=(2, 3))
        
        # Compute sum of the original block along T for carry update
        s_block_sum = tl.sum(s_block, axis=0)
        carry += s_block_sum

def chunk_global_reversed_cumsum_vector(s: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    B, H, T, S = s.shape
    BS = 32  # Fixed spatial block size
    
    # Initialize output tensor
    z = torch.empty_like(s, dtype=dtype)
    
    # Determine BT (time block size) using a simple heuristic; can be autotuned
    if T >= 2048:
        BT = 256
    elif T >= 1024:
        BT = 128
    elif T >= 512:
        BT = 64
    elif T >= 256:
        BT = 32
    else:
        BT = 16
    
    # Number of spatial blocks
    num_S_blocks = (S + BS - 1) // BS
    
    # Grid configuration
    grid = (B, H, num_S_blocks)
    
    # Launch kernel
    chunk_global_reversed_cumsum_vector_kernel[grid](
        s, z,
        B, H, T, S,
        s.stride(0), s.stride(1), s.stride(2), s.stride(3),
        z.stride(0), z.stride(1), z.stride(2), z.stride(3),
        BT=BT, BS=BS
    )
    
    return z
