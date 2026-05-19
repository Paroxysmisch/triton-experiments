import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    output_ptr,  # Pointer to output tensor
    input_ptr,   # Pointer to input tensor
    H,           # Number of heads
    T,           # Sequence length
    stride_b,    # Stride for batch dimension
    stride_h,    # Stride for head dimension
    stride_t,    # Stride for sequence dimension
    BT: tl.constexpr,  # Block size for T dimension
):
    # Get program ID for the current block
    pid = tl.program_id(0)
    
    # Calculate batch and head indices
    batch_idx = pid // H
    head_idx = pid % H
    
    # Calculate base offset for current (batch, head) pair
    base_offset = batch_idx * stride_b + head_idx * stride_h
    
    # Initialize accumulator
    b_z = 0.0
    
    # Iterate over sequence dimension in reverse order
    for t_block in range((T + BT - 1) // BT):
        # Calculate start and end positions for current block
        t_start = T - (t_block + 1) * BT
        t_end = T - t_block * BT
        
        # Handle boundary conditions
        t_start = tl.max(t_start, 0)
        
        # Create offsets for current block
        offs = base_offset + tl.arange(0, min(BT, t_end - t_start)) * stride_t + t_start * stride_t
        
        # Load input values
        block_mask = tl.arange(0, BT) < (t_end - t_start)
        x = tl.load(input_ptr + offs, mask=block_mask, other=0.0)
        
        # Compute block sum and update accumulator
        block_sum = tl.sum(x, axis=0)
        b_z = b_z + block_sum
        
        # Compute cumulative sum for current block
        cumsum = b_z - tl.cumsum(x, axis=0)
        
        # Store results
        tl.store(output_ptr + offs, cumsum, mask=block_mask)

def chunk_global_reversed_cumsum_scalar(x: torch.Tensor) -> torch.Tensor:
    """
    Compute reversed cumulative sum for a 3D tensor along the last dimension.
    
    Args:
        x: Input tensor of shape (B, H, T)
        
    Returns:
        Output tensor of shape (B, H, T) containing reversed cumulative sums
    """
    assert x.dim() == 3, "Input tensor must be 3-dimensional (B, H, T)"
    B, H, T = x.shape
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate strides
    stride_b = x.stride(0)
    stride_h = x.stride(1)
    stride_t = x.stride(2)
    
    # Define block size for T dimension
    BT = min(128, triton.next_power_of_2(T))
    
    # Launch kernel
    grid = (B * H,)
    chunk_global_reversed_cumsum_scalar_kernel[grid](
        output,
        x,
        H,
        T,
        stride_b,
        stride_h,
        stride_t,
        BT,
    )
    
    return output
