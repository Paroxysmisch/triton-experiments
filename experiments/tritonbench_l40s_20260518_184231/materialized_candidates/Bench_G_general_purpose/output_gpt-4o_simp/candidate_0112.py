import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(KV_nope_ptr, KV_rope_ptr, O_nope_ptr, O_rope_ptr, DestLoc_ptr,
                                  batch_stride, head_stride, dim_stride, seq_len,
                                  BLOCK_SIZE: tl.constexpr):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Compute the index for each element in the sequence
    index = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we do not go out of bounds
    mask = index < seq_len
    
    # Load destination indices
    dest_index = tl.load(DestLoc_ptr + index, mask=mask, other=0)
    
    # Compute the offsets for each tensor
    offset_kv_nope = index * dim_stride
    offset_kv_rope = index * dim_stride
    offset_o_nope = dest_index * dim_stride
    offset_o_rope = dest_index * dim_stride
    
    # Load data from source tensors
    kv_nope_val = tl.load(KV_nope_ptr + offset_kv_nope, mask=mask)
    kv_rope_val = tl.load(KV_rope_ptr + offset_kv_rope, mask=mask)
    
    # Store data to destination tensors
    tl.store(O_nope_ptr + offset_o_nope, kv_nope_val, mask=mask)
    tl.store(O_rope_ptr + offset_o_rope, kv_rope_val, mask=mask)

def destindex_copy_kv(KV_nope, KV_rope, O_nope, O_rope, DestLoc, seq_len, block_size=128):
    # Ensure that the dimensions are aligned to the next power of two
    assert KV_nope.shape == KV_rope.shape
    assert O_nope.shape == O_rope.shape
    assert KV_nope.shape[0] == DestLoc.shape[0]
    
    # Calculate strides
    batch_stride = KV_nope.stride(0)
    head_stride = KV_nope.stride(1)
    dim_stride = KV_nope.stride(2)
    
    # Launch the kernel
    grid = (triton.cdiv(seq_len, block_size),)
    _fwd_kernel_destindex_copy_kv[grid](
        KV_nope, KV_rope, O_nope, O_rope, DestLoc,
        batch_stride, head_stride, dim_stride, seq_len,
        BLOCK_SIZE=block_size
    )

# Example usage:
# KV_nope, KV_rope, O_nope, O_rope, DestLoc should be Triton-compatible tensors
# seq_len is the length of the sequence
# destindex_copy_kv(KV_nope, KV_rope, O_nope, O_rope, DestLoc, seq_len)
