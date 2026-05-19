import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K_ptr, DestLoc_ptr, Out_ptr, Out_scale_ptr,
    batch_size, head_size, head_dim, group_size,
    BLOCK_GROUP_NUM, BLOCK_GROUP_DIM,
    **meta
):
    # Compute the index of the current batch and head
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Compute the start of the group for this block
    group_start = tl.program_id(2) * BLOCK_GROUP_DIM
    
    # Initialize pointers for this batch and head
    K_ptr = K_ptr + (batch_idx * head_size + head_idx) * head_dim
    DestLoc_ptr = DestLoc_ptr + (batch_idx * head_size + head_idx) * head_dim
    
    # Load data for this group
    offsets = group_start + tl.arange(0, BLOCK_GROUP_DIM)
    K = tl.load(K_ptr + offsets, mask=offsets < head_dim, other=0.0)
    
    # Compute absolute max for scaling
    abs_max = tl.max(tl.abs(K), axis=0)
    
    # Compute scaling factor
    scale = abs_max / 127.0
    scale = tl.where(abs_max != 0, scale, 1.0)  # Avoid division by zero
    
    # Quantize data
    K_quantized = tl.cast(K / scale, tl.int8)
    
    # Store quantized data to the output tensor
    dest_index = tl.load(DestLoc_ptr + offsets, mask=offsets < head_dim, other=0)
    tl.store(Out_ptr + dest_index, K_quantized, mask=offsets < head_dim)
    
    # Store the scale factor
    tl.store(Out_scale_ptr + tl.program_id(2), scale, mask=offsets < head_dim)


def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale, group_size, BLOCK_GROUP_NUM, BLOCK_GROUP_DIM):
    batch_size, head_size, head_dim = K.shape
    
    # Ensure the head dimension is divisible by the group size
    assert head_dim % group_size == 0, "head_dim must be divisible by group_size"
    
    # Calculate grid size
    grid = (batch_size, head_size, head_dim // BLOCK_GROUP_DIM)
    
    # Launch the Triton kernel
    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K, DestLoc, Out, Out_scale,
        batch_size, head_size, head_dim, group_size,
        BLOCK_GROUP_NUM, BLOCK_GROUP_DIM
    )
