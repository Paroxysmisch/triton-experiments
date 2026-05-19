import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    # Pointers to input/output tensors
    s_ptr,          # Input tensor pointer [B, H, T]
    o_ptr,          # Output tensor pointer [B, H, T]
    # Dimensions
    B,              # Batch size
    H,              # Number of heads
    T,              # Sequence length
    BT,             # Block size for T dimension
    stride_b,       # Stride for batch dimension
    stride_h,       # Stride for head dimension
    stride_t,       # Stride for sequence dimension
    BLOCK_SIZE: tl.constexpr,  # Static block size
):
    # Compute the program ID
    pid = tl.program_id(0)
    
    # Calculate batch and head indices
    b_idx = pid // H
    h_idx = pid % H
    
    # Base pointer offset for current (b, h)
    base_offset = b_idx * stride_b + h_idx * stride_h
    
    # Initialize running sum
    running_sum = 0.0
    
    # Process chunks from right to left
    for t_start in range(T - BT, -BT, -BT):
        t_end = min(t_start + BT, T)
        t_start = max(t_start, 0)
        
        # Load offsets for current chunk
        offsets = base_offset + tl.arange(0, BLOCK_SIZE) * stride_t
        mask = tl.arange(0, BLOCK_SIZE) < (t_end - t_start)
        
        # Load chunk data
        chunk = tl.load(s_ptr + offsets + t_start * stride_t, mask=mask, other=0.0)
        
        # Compute cumsum within chunk (reversed)
        chunk_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for i in range(BLOCK_SIZE-1, -1, -1):
            if i < (t_end - t_start):
                running_sum += chunk[i]
                chunk_sum[i] = running_sum
        
        # Store results
        tl.store(o_ptr + offsets + t_start * stride_t, chunk_sum, mask=mask)

def chunk_global_reversed_cumsum_scalar(s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Compute reversed cumulative sum along the last dimension using Triton.
    
    Args:
        s: Input tensor of shape (B, H, T)
        block_size: Block size for processing chunks
    
    Returns:
        torch.Tensor: Output tensor of shape (B, H, T) containing reversed cumsum
    """
    assert len(s.shape) == 3, "Input tensor must be 3D (B, H, T)"
    B, H, T = s.shape
    
    # Create output tensor
    o = torch.empty_like(s)
    
    # Calculate strides
    stride_b, stride_h, stride_t = s.stride()
    
    # Launch kernel
    grid = (B * H,)
    chunk_global_reversed_cumsum_scalar_kernel[grid](
        s_ptr=s,
        o_ptr=o,
        B=B,
        H=H,
        T=T,
        BT=block_size,
        stride_b=stride_b,
        stride_h=stride_h,
        stride_t=stride_t,
        BLOCK_SIZE=block_size,
    )
    
    return o
