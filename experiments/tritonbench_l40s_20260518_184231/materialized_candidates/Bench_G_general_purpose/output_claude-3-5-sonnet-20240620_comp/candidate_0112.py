import triton
import triton.language as tl
import torch

# Constants for block sizes - power of 2 for optimal performance
BLOCK_DMODEL_NOPE = 128
BLOCK_DMODEL_ROPE = 128

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    # Pointers to tensors
    KV_nope_ptr, KV_rope_ptr,
    O_nope_ptr, O_rope_ptr,
    DestLoc_ptr,
    # Dimensions and strides
    batch_size, num_heads, head_dim_nope, head_dim_rope,
    KV_nope_batch_stride, KV_nope_head_stride,
    KV_rope_batch_stride, KV_rope_head_stride,
    O_nope_batch_stride, O_nope_head_stride,
    O_rope_batch_stride, O_rope_head_stride,
    BLOCK_DMODEL_NOPE: tl.constexpr, BLOCK_DMODEL_ROPE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate batch and head indices
    batch_idx = pid // num_heads
    head_idx = pid % num_heads
    
    # Compute base pointers for this batch and head
    KV_nope_offset = batch_idx * KV_nope_batch_stride + head_idx * KV_nope_head_stride
    KV_rope_offset = batch_idx * KV_rope_batch_stride + head_idx * KV_rope_head_stride
    O_nope_offset = batch_idx * O_nope_batch_stride + head_idx * O_nope_head_stride
    O_rope_offset = batch_idx * O_rope_batch_stride + head_idx * O_rope_head_stride
    
    # Create offsets for parallel processing
    offs_nope = tl.arange(0, BLOCK_DMODEL_NOPE)
    offs_rope = tl.arange(0, BLOCK_DMODEL_ROPE)
    
    # Load destination index for this batch and head
    dest_idx = tl.load(DestLoc_ptr + batch_idx * num_heads + head_idx)
    
    # Mask for bounds checking
    mask_nope = offs_nope < head_dim_nope
    mask_rope = offs_rope < head_dim_rope
    
    # Load from source tensors
    x_nope = tl.load(KV_nope_ptr + KV_nope_offset + offs_nope, mask=mask_nope)
    x_rope = tl.load(KV_rope_ptr + KV_rope_offset + offs_rope, mask=mask_rope)
    
    # Calculate destination offsets using dest_idx
    O_nope_dest_offset = O_nope_offset + dest_idx * head_dim_nope
    O_rope_dest_offset = O_rope_offset + dest_idx * head_dim_rope
    
    # Store to destination tensors
    tl.store(O_nope_ptr + O_nope_dest_offset + offs_nope, x_nope, mask=mask_nope)
    tl.store(O_rope_ptr + O_rope_dest_offset + offs_rope, x_rope, mask=mask_rope)

def destindex_copy_kv(KV_nope, KV_rope, DestLoc, O_nope, O_rope):
    # Input validation
    assert KV_nope.shape[:-1] == O_nope.shape[:-1], "Input shapes must match"
    assert KV_rope.shape[:-1] == O_rope.shape[:-1], "Input shapes must match"
    assert len(DestLoc.shape) == 2, "DestLoc must be 2D tensor"
    
    # Extract dimensions
    batch_size, num_heads = KV_nope.shape[:2]
    head_dim_nope = KV_nope.shape[-1]
    head_dim_rope = KV_rope.shape[-1]
    
    # Compute strides
    KV_nope_batch_stride = KV_nope.stride(0)
    KV_nope_head_stride = KV_nope.stride(1)
    KV_rope_batch_stride = KV_rope.stride(0)
    KV_rope_head_stride = KV_rope.stride(1)
    
    O_nope_batch_stride = O_nope.stride(0)
    O_nope_head_stride = O_nope.stride(1)
    O_rope_batch_stride = O_rope.stride(0)
    O_rope_head_stride = O_rope.stride(1)
    
    # Launch kernel
    grid = (batch_size * num_heads,)
    _fwd_kernel_destindex_copy_kv[grid](
        KV_nope, KV_rope,
        O_nope, O_rope,
        DestLoc,
        batch_size, num_heads, head_dim_nope, head_dim_rope,
        KV_nope_batch_stride, KV_nope_head_stride,
        KV_rope_batch_stride, KV_rope_head_stride,
        O_nope_batch_stride, O_nope_head_stride,
        O_rope_batch_stride, O_rope_head_stride,
        BLOCK_DMODEL_NOPE=BLOCK_DMODEL_NOPE,
        BLOCK_DMODEL_ROPE=BLOCK_DMODEL_ROPE,
        num_warps=4,
        num_stages=3,
    )
    
    return O_nope, O_rope
