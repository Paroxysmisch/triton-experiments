import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    # Pointers to tensors
    KV_nope_ptr, KV_rope_ptr,    # Source tensors
    O_nope_ptr, O_rope_ptr,      # Destination tensors
    DestLoc_ptr,                 # Indices for destination locations
    # Dimensions and strides
    batch_size, num_heads, seq_len, dim,
    KV_batch_stride, KV_head_stride, KV_seq_stride,
    O_batch_stride, O_head_stride, O_seq_stride,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate current position
    pid = tl.program_id(0)
    batch_idx = pid // (num_heads * seq_len)
    head_idx = (pid % (num_heads * seq_len)) // seq_len
    seq_idx = pid % seq_len

    # Early exit if out of bounds
    if batch_idx >= batch_size or head_idx >= num_heads:
        return

    # Calculate offsets for current position
    kv_offset = batch_idx * KV_batch_stride + head_idx * KV_head_stride + seq_idx * KV_seq_stride
    
    # Load destination index for current position
    dest_idx = tl.load(DestLoc_ptr + seq_idx)
    
    # Calculate destination offset
    o_offset = batch_idx * O_batch_stride + head_idx * O_head_stride + dest_idx * O_seq_stride

    # Create block pointers for dimension
    dim_offsets = tl.arange(0, BLOCK_SIZE)
    mask = dim_offsets < dim

    # Load from source tensors
    kv_nope = tl.load(KV_nope_ptr + kv_offset + dim_offsets, mask=mask)
    kv_rope = tl.load(KV_rope_ptr + kv_offset + dim_offsets, mask=mask)

    # Store to destination tensors
    tl.store(O_nope_ptr + o_offset + dim_offsets, kv_nope, mask=mask)
    tl.store(O_rope_ptr + o_offset + dim_offsets, kv_rope, mask=mask)

def destindex_copy_kv(KV_nope, KV_rope, DestLoc, O_nope, O_rope):
    """
    Wrapper function for the destindex_copy_kv kernel.
    
    Args:
        KV_nope: Source tensor without positional encoding
        KV_rope: Source tensor with positional encoding
        DestLoc: Tensor containing destination indices
        O_nope: Destination tensor for values without positional encoding
        O_rope: Destination tensor for values with positional encoding
    """
    # Get dimensions
    batch_size, num_heads, seq_len, dim = KV_nope.shape
    
    # Calculate next power of 2 for dimension
    BLOCK_SIZE = triton.next_power_of_2(dim)
    
    # Calculate strides
    KV_batch_stride = KV_nope.stride(0)
    KV_head_stride = KV_nope.stride(1)
    KV_seq_stride = KV_nope.stride(2)
    
    O_batch_stride = O_nope.stride(0)
    O_head_stride = O_nope.stride(1)
    O_seq_stride = O_nope.stride(2)

    # Launch kernel
    grid = (batch_size * num_heads * seq_len,)
    _fwd_kernel_destindex_copy_kv[grid](
        KV_nope, KV_rope,
        O_nope, O_rope,
        DestLoc,
        batch_size, num_heads, seq_len, dim,
        KV_batch_stride, KV_head_stride, KV_seq_stride,
        O_batch_stride, O_head_stride, O_seq_stride,
        BLOCK_SIZE=BLOCK_SIZE
    )
