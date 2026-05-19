import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    K_ptr, Out_ptr, DestLoc_ptr,
    stride_kb, stride_kh, stride_kd,
    stride_ob, stride_oh, stride_od,
    BLOCK_HEAD: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    n_heads, d_head, seq_len
):
    # Get program ID
    cur_index = tl.program_id(0)
    
    # Load destination location for current index
    dest_idx = tl.load(DestLoc_ptr + cur_index)
    
    # Calculate offsets for heads and depth dimensions
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Mask for bounds checking
    mask_h = offs_h < n_heads
    mask_d = offs_d < d_head
    
    # Iterate over heads in blocks
    for h in range(0, n_heads, BLOCK_HEAD):
        h_offs = h + offs_h
        
        # Iterate over depth dimension in blocks
        for d in range(0, d_head, BLOCK_DMODEL):
            d_offs = d + offs_d
            
            # Calculate source and destination addresses
            k_ptrs = K_ptr + dest_idx * stride_kb + h_offs[:, None] * stride_kh + d_offs[None, :] * stride_kd
            o_ptrs = Out_ptr + cur_index * stride_ob + h_offs[:, None] * stride_oh + d_offs[None, :] * stride_od
            
            # Load from source with masking
            mask = mask_h[:, None] & mask_d[None, :]
            vals = tl.load(k_ptrs, mask=mask)
            
            # Store to destination with masking
            tl.store(o_ptrs, vals, mask=mask)

def destindex_copy_kv(k: torch.Tensor, out: torch.Tensor, dest_loc: torch.Tensor):
    """
    Wrapper function for the destindex copy kernel
    Args:
        k: Input tensor of shape [batch_size, n_heads, d_head]
        out: Output tensor of shape [seq_len, n_heads, d_head]
        dest_loc: Destination indices tensor of shape [seq_len]
    """
    batch_size, n_heads, d_head = k.shape
    seq_len = out.shape[0]
    
    # Compute block sizes as powers of 2
    BLOCK_HEAD = min(triton.next_power_of_2(n_heads), 32)
    BLOCK_DMODEL = min(triton.next_power_of_2(d_head), 64)
    
    # Get strides for both tensors
    stride_kb, stride_kh, stride_kd = k.stride()
    stride_ob, stride_oh, stride_od = out.stride()
    
    # Launch kernel
    grid = (seq_len,)
    
    _fwd_kernel_destindex_copy_kv[grid](
        k, out, dest_loc,
        stride_kb, stride_kh, stride_kd,
        stride_ob, stride_oh, stride_od,
        BLOCK_HEAD, BLOCK_DMODEL,
        n_heads, d_head, seq_len
    )
    
    return out
