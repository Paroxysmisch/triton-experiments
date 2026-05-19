import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K_ptr,          # pointer to input tensor [B, H, L, D]
    Dest_loc_ptr,   # pointer to destination indices [B, L]
    Out_ptr,        # pointer to output tensor [B, H, L, D]
    seq_len,        # sequence length
    head_num,       # number of heads
    head_dim,       # dimension per head
    stride_k_b,     # stride for batch in K
    stride_k_h,     # stride for head in K
    stride_k_l,     # stride for length in K
    stride_dest_b,  # stride for batch in Dest_loc
    stride_out_b,   # stride for batch in Out
    stride_out_h,   # stride for head in Out
    stride_out_l,   # stride for length in Out
    BLOCK_SEQ: tl.constexpr,    # block size for sequence dimension
    BLOCK_HEAD: tl.constexpr,   # block size for head dimension
):
    # Program ID
    pid = tl.program_id(0)
    bid = pid // ((seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ)
    lid = pid % ((seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ)

    # Compute the sequence offset
    seq_offset = lid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    seq_mask = seq_offset < seq_len

    # Load destination indices for current batch and sequence positions
    dest_idx = tl.load(Dest_loc_ptr + bid * stride_dest_b + seq_offset, mask=seq_mask)

    # Loop over heads
    for h in range(0, head_num, BLOCK_HEAD):
        head_offset = tl.arange(0, BLOCK_HEAD)
        head_mask = head_offset + h < head_num
        
        # Combined mask for sequence and head
        mask = seq_mask[:, None] & head_mask[None, :]

        # Load from source positions
        k_ptrs = K_ptr + bid * stride_k_b + \
                 (h + head_offset[None, :]) * stride_k_h + \
                 dest_idx[:, None] * stride_k_l
        
        # Load values from K
        k_vals = tl.load(k_ptrs, mask=mask, other=0.0)

        # Store to output positions
        out_ptrs = Out_ptr + bid * stride_out_b + \
                   (h + head_offset[None, :]) * stride_out_h + \
                   seq_offset[:, None] * stride_out_l
        
        # Store values to Out
        tl.store(out_ptrs, k_vals, mask=mask)

def destindex_copy_kv(k: torch.Tensor, dest_loc: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the Triton kernel that performs index-based copying.
    
    Args:
        k: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
        dest_loc: Destination indices tensor of shape [batch_size, seq_len]
    
    Returns:
        Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, _, head_dim = k.shape
    seq_len = dest_loc.shape[1]
    
    # Create output tensor
    out = torch.empty((batch_size, num_heads, seq_len, head_dim),
                     device=k.device, dtype=k.dtype)
    
    # Calculate strides
    stride_k_b = k.stride(0)
    stride_k_h = k.stride(1)
    stride_k_l = k.stride(2)
    stride_dest_b = dest_loc.stride(0)
    stride_out_b = out.stride(0)
    stride_out_h = out.stride(1)
    stride_out_l = out.stride(2)
    
    # Define block sizes
    BLOCK_SEQ = 32
    BLOCK_HEAD = 8
    
    # Calculate grid size
    grid = (batch_size * ((seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ),)
    
    # Launch kernel
    _fwd_kernel_destindex_copy_kv[grid](
        k, dest_loc, out,
        seq_len, num_heads, head_dim,
        stride_k_b, stride_k_h, stride_k_l,
        stride_dest_b,
        stride_out_b, stride_out_h, stride_out_l,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_HEAD=BLOCK_HEAD,
    )
    
    return out
