import triton
import triton.language as tl
import torch

@triton.jit
def chunk_global_reversed_cumsum_vector_kernel(
    # Pointers to tensors
    z_ptr, s_ptr,
    # Dimensions
    B, H, T, S,
    # Strides for the tensors
    s_stride_b, s_stride_h, s_stride_t, s_stride_s,
    z_stride_b, z_stride_h, z_stride_t, z_stride_s,
    # Block sizes
    BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr,
):
    # Get program ID
    pid_b = tl.program_id(0)  # Batch
    pid_h = tl.program_id(1)  # Head
    pid_t = tl.program_id(2)  # Time block
    
    # Initialize offsets
    b_offs = pid_b
    h_offs = pid_h
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    
    # Time block starting position (working backwards)
    t_start = T - (pid_t + 1) * BLOCK_T
    
    # Generate offset arrays for the spatial dimension
    s_offs = tl.arange(0, BLOCK_S)
    
    # Base pointers for current batch and head
    s_base_ptr = s_ptr + b_offs * s_stride_b + h_offs * s_stride_h
    z_base_ptr = z_ptr + b_offs * z_stride_b + h_offs * z_stride_h
    
    # Iterate through time steps in reverse
    for t in range(BLOCK_T):
        t_pos = t_start + t
        if t_pos >= 0:  # Boundary check
            # Load input block
            s_ptrs = s_base_ptr + t_pos * s_stride_t + s_offs * s_stride_s
            block_s = tl.load(s_ptrs, mask=s_offs < S, other=0.0)
            
            # Update accumulator
            acc += block_s
            
            # Store result
            z_ptrs = z_base_ptr + t_pos * z_stride_t + s_offs * z_stride_s
            tl.store(z_ptrs, acc, mask=s_offs < S)

def chunk_global_reversed_cumsum_vector(s: torch.Tensor) -> torch.Tensor:
    """
    Compute reversed cumulative sum across time dimension of a 4D tensor.
    
    Args:
        s: Input tensor of shape [B, H, T, S]
        
    Returns:
        z: Output tensor of same shape with reversed cumsum along time dimension
    """
    # Get dimensions
    B, H, T, S = s.shape
    
    # Create output tensor
    z = torch.empty_like(s)
    
    # Calculate strides
    s_stride_b, s_stride_h, s_stride_t, s_stride_s = s.stride()
    z_stride_b, z_stride_h, z_stride_t, z_stride_s = z.stride()
    
    # Define block sizes
    BLOCK_T = 32
    BLOCK_S = 32
    
    # Calculate grid dimensions
    grid = (
        B,                          # Batch dimension
        H,                          # Head dimension
        triton.cdiv(T, BLOCK_T),   # Time blocks
    )
    
    # Launch kernel
    chunk_global_reversed_cumsum_vector_kernel[grid](
        z, s,
        B, H, T, S,
        s_stride_b, s_stride_h, s_stride_t, s_stride_s,
        z_stride_b, z_stride_h, z_stride_t, z_stride_s,
        BLOCK_T=BLOCK_T,
        BLOCK_S=BLOCK_S,
    )
    
    return z
