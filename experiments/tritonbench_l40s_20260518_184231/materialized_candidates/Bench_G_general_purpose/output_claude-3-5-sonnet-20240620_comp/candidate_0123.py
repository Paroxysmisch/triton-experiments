import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_vector_kernel(
    s_ptr,          # pointer to input tensor [B, H, T, S]
    z_ptr,          # pointer to output tensor [B, H, T, S]
    B,              # batch size
    H,              # number of heads
    T,              # sequence length
    S,              # feature dimension
    stride_b,       # stride for batch dimension
    stride_h,       # stride for head dimension
    stride_t,       # stride for time dimension
    stride_s,       # stride for feature dimension
    BT: tl.constexpr,  # BLOCK SIZE along T dimension
    BS: tl.constexpr,  # BLOCK SIZE along S dimension
):
    # Program ID
    pid = tl.program_id(0)  # batch * heads combinations
    bid = pid // H
    hid = pid % H

    # Initialize pointers to current batch and head
    base_s_ptr = s_ptr + bid * stride_b + hid * stride_h
    base_z_ptr = z_ptr + bid * stride_b + hid * stride_h

    # Initialize accumulator for reversed cumsum
    acc = tl.zeros([BS], dtype=tl.float32)
    
    # Iterate backwards over time blocks
    for t in range(tl.cdiv(T, BT) - 1, -1, -1):
        t_offset = t * BT
        
        # Create block pointers
        offs_t = tl.arange(0, BT)
        offs_s = tl.arange(0, BS)
        
        # Compute actual time indices and mask for boundary check
        t_idx = t_offset + offs_t
        t_mask = t_idx < T
        
        # Load input block
        s_ptrs = base_s_ptr + t_idx[:, None] * stride_t + offs_s[None, :] * stride_s
        b_s = tl.load(s_ptrs, mask=t_mask[:, None], other=0.0)
        
        # Compute cumsum for current block
        b_z = tl.sum(b_s, axis=1) + acc
        
        # Store results
        z_ptrs = base_z_ptr + t_idx * stride_t
        tl.store(z_ptrs, b_z, mask=t_mask)
        
        # Update accumulator
        acc += tl.sum(b_s, axis=0)

# Python wrapper function
def chunk_global_reversed_cumsum_vector(s: torch.Tensor, dtype=None) -> torch.Tensor:
    if dtype is None:
        dtype = s.dtype
    
    # Extract dimensions
    B, H, T, S = s.shape
    
    # Set block sizes
    BS = 32  # spatial block size
    BT = 32  # temporal block size
    
    # Create output tensor
    z = torch.empty_like(s, dtype=dtype)
    
    # Calculate strides
    stride_b = s.stride(0)
    stride_h = s.stride(1)
    stride_t = s.stride(2)
    stride_s = s.stride(3)
    
    # Launch kernel
    grid = (B * H,)  # One program per (batch, head) combination
    
    chunk_global_reversed_cumsum_vector_kernel[grid](
        s_ptr=s.data_ptr(),
        z_ptr=z.data_ptr(),
        B=B, H=H, T=T, S=S,
        stride_b=stride_b,
        stride_h=stride_h,
        stride_t=stride_t,
        stride_s=stride_s,
        BT=BT,
        BS=BS,
    )
    
    return z
