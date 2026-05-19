import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    s, o,
    stride_bh, stride_t,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr,
    BT: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate batch and head indices
    batch = pid // H
    head = pid % H
    
    # Initialize accumulator
    b_z = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over T dimension in reverse order
    for t in range((T + BT - 1) // BT - 1, -1, -1):
        # Calculate offsets
        offs = batch * stride_bh + head * stride_t + t * BT + tl.arange(0, BT)
        mask = offs < (batch * stride_bh + head * stride_t + T)
        
        # Load input values
        x = tl.load(s + offs, mask=mask, other=0.0)
        
        # Compute block sum and update accumulator
        block_sum = tl.sum(x)
        b_z += block_sum
        
        # Compute cumulative sum for the block
        cumsum = b_z - tl.cumsum(x)
        
        # Store results
        tl.store(o + offs, cumsum, mask=mask)

def chunk_global_reversed_cumsum_scalar(s: torch.Tensor) -> torch.Tensor:
    assert s.dim() == 3, "Input tensor must be 3-dimensional (B, H, T)"
    B, H, T = s.shape
    
    # Output tensor
    o = torch.empty_like(s)
    
    # Kernel parameters
    BT = 1024  # Can be tuned for performance
    grid = (B * H,)
    
    # Launch kernel
    chunk_global_reversed_cumsum_scalar_kernel[grid](
        s, o,
        s.stride(0), s.stride(2),
        B=B, H=H, T=T,
        BT=BT
    )
    
    return o
