import triton
import triton.language as tl
import torch
import math

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K,  # Source tensor [batch, seq_len, head_num, head_dim]
    Out,  # Output tensor [batch, seq_len, head_num, head_dim] 
    Dest_loc,  # Destination indices [seq_len]
    stride_k_b, stride_k_s, stride_k_h, stride_k_d,  # Strides for K tensor
    stride_o_b, stride_o_s, stride_o_h, stride_o_d,  # Strides for Out tensor
    head_num: tl.constexpr,  # Number of heads
    head_dim: tl.constexpr,  # Head dimension
    BLOCK_HEAD: tl.constexpr,  # Block size for heads (power of 2)
):
    # Get current position
    pid = tl.program_id(0)
    
    # Get source and destination indices
    dest_idx = tl.load(Dest_loc + pid)
    
    # Create offsets for heads
    offs_h = tl.arange(0, BLOCK_HEAD)
    mask_h = offs_h < head_num
    
    # Create offsets for depth dimension
    offs_d = tl.arange(0, head_dim)
    
    # Compute base pointers for current sequence position
    k_base = K + pid * stride_k_s
    o_base = Out + dest_idx * stride_o_s
    
    # For each head
    for h in range(0, head_num, BLOCK_HEAD):
        # Compute pointers with head offset
        k_ptrs = k_base + (offs_h + h) * stride_k_h[:, None] + offs_d * stride_k_d
        o_ptrs = o_base + (offs_h + h) * stride_o_h[:, None] + offs_d * stride_o_d
        
        # Load from source with mask
        x = tl.load(k_ptrs, mask=mask_h[:, None])
        
        # Store to destination with mask
        tl.store(o_ptrs, x, mask=mask_h[:, None])

def destindex_copy_kv(k: torch.Tensor, out: torch.Tensor, dest_loc: torch.Tensor):
    """
    Wrapper function for the Triton kernel that copies values from K to Out based on destination indices.
    
    Args:
        k: Source tensor of shape [batch, seq_len, head_num, head_dim]
        out: Output tensor of shape [batch, seq_len, head_num, head_dim]
        dest_loc: Destination indices tensor of shape [seq_len]
    """
    batch, seq_len, head_num, head_dim = k.shape
    assert out.shape == k.shape, "Input and output tensors must have same shape"
    assert dest_loc.shape == (seq_len,), "Destination indices must match sequence length"
    
    # Calculate strides
    stride_k = k.stride()
    stride_o = out.stride()
    
    # Calculate BLOCK_HEAD as next power of 2 of head_num
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    
    # Launch kernel
    grid = (seq_len,)
    _fwd_kernel_destindex_copy_kv[grid](
        k, out, dest_loc,
        stride_k[0], stride_k[1], stride_k[2], stride_k[3],  # K strides
        stride_o[0], stride_o[1], stride_o[2], stride_o[3],  # Out strides
        head_num=head_num,
        head_dim=head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
    )
    
    return out
