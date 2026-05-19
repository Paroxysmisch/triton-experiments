import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_cumsum_scalar_kernel(
    # Pointers to input/output tensors
    s_ptr,          # Input tensor pointer [B, H, T]
    o_ptr,          # Output tensor pointer [B, H, T]
    # Dimensions
    batch_size,     # Batch size (B)
    num_heads,      # Number of heads (H) 
    seq_len,        # Sequence length (T)
    BT,             # Block size for time dimension
    stride_b,       # Stride for batch dimension
    stride_h,       # Stride for head dimension
    stride_t,       # Stride for time dimension
    BLOCK_SIZE: tl.constexpr,  # Static block size
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate batch and head indices
    batch_idx = pid // num_heads
    head_idx = pid % num_heads
    
    # Calculate base offset for this (batch, head)
    base_offset = batch_idx * stride_b + head_idx * stride_h
    
    # Initialize running sum
    running_sum = 0.0
    
    # Process sequence in chunks of size BT
    for t_start in range(0, seq_len, BT):
        # Calculate actual block size (handle boundary)
        curr_block_size = min(BT, seq_len - t_start)
        
        # Create block pointers
        s_block_ptr = tl.make_block_ptr(
            base=s_ptr + base_offset,
            shape=(seq_len,),
            strides=(stride_t,),
            offsets=(t_start,),
            block_shape=(curr_block_size,),
            order=(0,)
        )
        
        o_block_ptr = tl.make_block_ptr(
            base=o_ptr + base_offset,
            shape=(seq_len,),
            strides=(stride_t,),
            offsets=(t_start,),
            block_shape=(curr_block_size,),
            order=(0,)
        )
        
        # Load block
        x = tl.load(s_block_ptr)
        
        # Add running sum to all elements
        x = x + running_sum
        
        # Compute cumsum within block
        x = tl.cumsum(x, axis=0)
        
        # Store result
        tl.store(o_block_ptr, x)
        
        # Update running sum with last element
        running_sum = tl.sum(x[-1:])

def chunk_global_cumsum_scalar(s: torch.Tensor, dtype=None):
    """
    Wrapper function for chunk-based cumulative sum kernel.
    
    Args:
        s: Input tensor of shape [B, H, T]
        dtype: Optional dtype for output tensor
    
    Returns:
        z: Output tensor of shape [B, H, T] containing cumulative sums
    """
    if dtype is None:
        dtype = s.dtype
        
    # Get dimensions
    batch_size, num_heads, seq_len = s.shape
    
    # Create output tensor
    z = torch.empty_like(s, dtype=dtype)
    
    # Calculate strides
    stride_b = s.stride(0)
    stride_h = s.stride(1)
    stride_t = s.stride(2)
    
    # Calculate optimal block size (power of 2)
    BT = min(triton.next_power_of_2(seq_len), 1024)
    
    # Launch kernel
    grid = (batch_size * num_heads,)
    chunk_global_cumsum_scalar_kernel[grid](
        s_ptr=s,
        o_ptr=z,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        BT=BT,
        stride_b=stride_b,
        stride_h=stride_h,
        stride_t=stride_t,
        BLOCK_SIZE=BT,
    )
    
    return z
