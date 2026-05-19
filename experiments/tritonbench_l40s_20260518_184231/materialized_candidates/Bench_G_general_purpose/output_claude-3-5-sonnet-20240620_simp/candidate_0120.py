import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_cumsum_scalar_kernel(
    o_ptr,          # Output tensor pointer
    s_ptr,          # Input tensor pointer
    running_total,  # Running total across blocks
    stride_b,       # Batch stride
    stride_h,       # Height stride
    stride_w,       # Width stride
    B,              # Batch size
    H,              # Height
    W,              # Width
    BT: tl.constexpr,  # Block size along the reduction dimension
):
    # Get program ID
    pid_b = tl.program_id(0)  # Batch
    pid_h = tl.program_id(1)  # Height
    pid_w = tl.program_id(2)  # Width block
    
    # Compute the starting offset for this block
    block_start = pid_w * BT
    
    # Load the running total from previous blocks
    prev_total = tl.load(running_total + pid_b * H * (W // BT) + pid_h * (W // BT) + pid_w)
    
    # Create offsets for this thread block
    offsets = block_start + tl.arange(0, BT)
    mask = offsets < W
    
    # Compute base pointer for this batch and height
    base_ptr = s_ptr + pid_b * stride_b + pid_h * stride_h
    
    # Load input values
    x = tl.load(base_ptr + offsets * stride_w, mask=mask, other=0.0)
    
    # Compute local cumsum
    local_cumsum = tl.cumsum(x)
    
    # Add previous total to maintain global cumsum
    output = local_cumsum + prev_total
    
    # Store results
    o_base_ptr = o_ptr + pid_b * stride_b + pid_h * stride_h
    tl.store(o_base_ptr + offsets * stride_w, output, mask=mask)
    
    # Update running total for next block
    if tl.sum(mask) > 0:
        last_valid_idx = tl.max(tl.where(mask, offsets, -1))
        new_total = tl.load(o_base_ptr + last_valid_idx * stride_w)
        tl.store(running_total + pid_b * H * (W // BT) + pid_h * (W // BT) + pid_w + 1, new_total)

def chunk_global_cumsum_scalar(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """
    Compute chunked global cumulative sum along the last dimension.
    
    Args:
        x: Input tensor of shape (B, H, W)
        block_size: Size of blocks for processing
        
    Returns:
        Tensor of same shape as input with cumulative sum computed
    """
    assert x.dim() == 3, "Input tensor must be 3D (B, H, W)"
    B, H, W = x.shape
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate number of blocks and create running totals tensor
    num_blocks = (W + block_size - 1) // block_size
    running_totals = torch.zeros((B, H, num_blocks + 1), 
                               dtype=x.dtype, 
                               device=x.device)
    
    # Launch kernel
    grid = (B, H, num_blocks)
    chunk_global_cumsum_scalar_kernel[grid](
        output,
        x,
        running_totals,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        B, H, W,
        BT=block_size,
    )
    
    return output
