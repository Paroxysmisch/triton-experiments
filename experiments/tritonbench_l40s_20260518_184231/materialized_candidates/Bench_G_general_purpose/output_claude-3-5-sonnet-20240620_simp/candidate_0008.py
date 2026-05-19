import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    # Pointers to tensors
    Dest_ptr, Dest_scale_ptr, Src_ptr, Dest_loc_ptr,
    # Dimensions and strides
    seq_len, head_dim, num_heads,
    Dest_batch_stride, Dest_head_stride, Dest_seq_stride,
    Src_batch_stride, Src_head_stride, Src_seq_stride,
    # Group size for quantization
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute indices for this program instance
    pid = tl.program_id(0)
    num_seq_heads = seq_len * num_heads
    seq_id = pid // num_heads
    head_id = pid % num_heads

    # Calculate offsets
    src_offset = seq_id * Src_seq_stride + head_id * Src_head_stride
    
    # Load destination location index
    dest_idx = tl.load(Dest_loc_ptr + seq_id)
    dest_offset = dest_idx * Dest_seq_stride + head_id * Dest_head_stride

    # Number of groups for this sequence
    num_groups = head_dim // GROUP_SIZE

    # Process each group
    for group_idx in range(num_groups):
        # Calculate group offsets
        group_start = group_idx * GROUP_SIZE
        
        # Load source data for this group
        src_vals = tl.load(Src_ptr + src_offset + tl.arange(0, GROUP_SIZE))
        
        # Compute scaling factor for quantization
        max_val = tl.max(tl.abs(src_vals))
        scale = max_val / 127.0  # For int8 quantization
        
        # Quantize the values
        quant_vals = tl.math.round(src_vals / scale)
        quant_vals = tl.math.clamp(quant_vals, -127.0, 127.0)
        
        # Store quantized values and scale
        tl.store(Dest_ptr + dest_offset + group_start, quant_vals.to(tl.int8))
        if tl.program_id(1) == 0:  # Only store scale once per group
            tl.store(Dest_scale_ptr + dest_idx * num_groups + group_idx, scale)

def destindex_copy_quantize_kv(dest_tensor, dest_scale, src_tensor, dest_loc):
    """
    Wrapper function for the quantization and copy kernel.
    
    Args:
        dest_tensor: Destination tensor for quantized values (int8)
        dest_scale: Destination tensor for scaling factors (float32)
        src_tensor: Source tensor (float32)
        dest_loc: Index mapping tensor
    """
    # Extract dimensions
    batch_size, seq_len, num_heads, head_dim = src_tensor.shape
    
    # Constants for the kernel
    GROUP_SIZE = 16
    BLOCK_SIZE = 32
    
    # Compute grid dimensions
    grid = (seq_len * num_heads,)
    
    # Launch kernel
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        dest_tensor, dest_scale, src_tensor, dest_loc,
        seq_len, head_dim, num_heads,
        dest_tensor.stride(0), dest_tensor.stride(2), dest_tensor.stride(1),
        src_tensor.stride(0), src_tensor.stride(2), src_tensor.stride(1),
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )
