import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K_ptr, Dest_loc_ptr, Out_ptr, Out_scale_ptr,
    seq_len, head_dim, group_size,
    K_batch_stride, K_head_stride,
    Out_batch_stride, Out_head_stride,
    BLOCK_GROUP_NUM: tl.constexpr, BLOCK_GROUP_DIM: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate batch and head indices
    batch_id = pid // (seq_len)
    seq_id = pid % seq_len
    
    # Load destination index
    dest_index = tl.load(Dest_loc_ptr + seq_id)
    
    # Calculate base pointers for current batch
    K_base = K_ptr + batch_id * K_batch_stride
    Out_base = Out_ptr + batch_id * Out_batch_stride
    
    # Process groups of elements
    for group_idx in range(0, head_dim, BLOCK_GROUP_DIM):
        # Create offsets for loading data
        offs_group = tl.arange(0, BLOCK_GROUP_DIM)
        mask_group = offs_group < (head_dim - group_idx)
        
        # Load source data
        k_offs = seq_id * K_head_stride + group_idx + offs_group
        k_data = tl.load(K_base + k_offs, mask=mask_group, other=0.0)
        
        # Compute absolute maximum for scaling
        max_val = tl.max(tl.abs(k_data), axis=0)
        scale = max_val / 127.0  # Scale to int8 range
        
        # Quantize data
        k_quantized = tl.where(mask_group, 
                             tl.math.round(k_data / (scale + 1e-5)),
                             0)
        
        # Store quantized data
        out_offs = dest_index * Out_head_stride + group_idx + offs_group
        tl.store(Out_base + out_offs, k_quantized.to(tl.int8), mask=mask_group)
        
        # Store scale factor
        if tl.program_id(1) == 0:  # Only store scale once per group
            scale_idx = dest_index * (head_dim // BLOCK_GROUP_DIM) + (group_idx // BLOCK_GROUP_DIM)
            tl.store(Out_scale_ptr + scale_idx, scale)

def destindex_copy_quantize_kv(k: torch.Tensor, 
                              dest_loc: torch.Tensor,
                              group_size: int = 64):
    """
    Wrapper function for KV quantization kernel
    
    Args:
        k: Input tensor of shape (batch, seq_len, head_dim)
        dest_loc: Destination indices tensor of shape (seq_len,)
        group_size: Size of groups for quantization
    
    Returns:
        Tuple of (quantized tensor, scale factors tensor)
    """
    batch, seq_len, head_dim = k.shape
    device = k.device
    
    # Ensure head dimension is divisible by group size
    assert head_dim % group_size == 0, f"Head dimension {head_dim} must be divisible by group size {group_size}"
    
    # Initialize output tensors
    out = torch.empty((batch, seq_len, head_dim), 
                     dtype=torch.int8, 
                     device=device)
    out_scale = torch.empty((batch, seq_len, head_dim // group_size),
                           dtype=torch.float32,
                           device=device)
    
    # Define kernel constants
    BLOCK_GROUP_NUM = 1
    BLOCK_GROUP_DIM = group_size
    
    # Launch kernel
    grid = (batch * seq_len,)
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        k, dest_loc, out, out_scale,
        seq_len, head_dim, group_size,
        k.stride(0), k.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_GROUP_NUM=BLOCK_GROUP_NUM,
        BLOCK_GROUP_DIM=BLOCK_GROUP_DIM
    )
    
    return out, out_scale
