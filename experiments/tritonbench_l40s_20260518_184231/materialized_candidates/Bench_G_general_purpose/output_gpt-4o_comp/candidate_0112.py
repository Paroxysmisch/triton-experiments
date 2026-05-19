import triton
import triton.language as tl

# Define constants for block sizes
BLOCK_DMODEL_NOPE = 128  # This should be a power of two
BLOCK_DMODEL_ROPE = 128  # This should be a power of two

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    KV_nope_ptr, KV_rope_ptr, DestLoc_ptr, O_nope_ptr, O_rope_ptr,
    batch_size, num_heads, head_dim_nope, head_dim_rope,
    stride_kv_batch, stride_kv_head, stride_kv_dmodel_nope,
    stride_kv_dmodel_rope, stride_o_batch, stride_o_head,
    stride_o_dmodel_nope, stride_o_dmodel_rope,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the position in the grid
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    dmodel_offset_nope = tl.arange(0, BLOCK_DMODEL_NOPE)
    dmodel_offset_rope = tl.arange(0, BLOCK_DMODEL_ROPE)

    # Compute base pointers for the current batch and head
    kv_nope_ptr = KV_nope_ptr + batch_id * stride_kv_batch + head_id * stride_kv_head
    kv_rope_ptr = KV_rope_ptr + batch_id * stride_kv_batch + head_id * stride_kv_head
    o_nope_ptr = O_nope_ptr + batch_id * stride_o_batch + head_id * stride_o_head
    o_rope_ptr = O_rope_ptr + batch_id * stride_o_batch + head_id * stride_o_head

    # Load destination indices
    dest_loc = tl.load(DestLoc_ptr + batch_id * num_heads + head_id)

    # Load data from KV_nope and KV_rope using the destination indices
    kv_nope_data = tl.load(kv_nope_ptr + dest_loc * stride_kv_dmodel_nope + dmodel_offset_nope, mask=dmodel_offset_nope < head_dim_nope)
    kv_rope_data = tl.load(kv_rope_ptr + dest_loc * stride_kv_dmodel_rope + dmodel_offset_rope, mask=dmodel_offset_rope < head_dim_rope)

    # Store data into O_nope and O_rope at the correct locations
    tl.store(o_nope_ptr + dest_loc * stride_o_dmodel_nope + dmodel_offset_nope, kv_nope_data, mask=dmodel_offset_nope < head_dim_nope)
    tl.store(o_rope_ptr + dest_loc * stride_o_dmodel_rope + dmodel_offset_rope, kv_rope_data, mask=dmodel_offset_rope < head_dim_rope)

def destindex_copy_kv(KV_nope, KV_rope, DestLoc, O_nope, O_rope):
    # Validate shapes
    assert KV_nope.shape == KV_rope.shape
    assert O_nope.shape == O_rope.shape
    assert KV_nope.shape == O_nope.shape
    assert DestLoc.shape[0] == KV_nope.shape[0] * KV_nope.shape[1]

    batch_size, num_heads, head_dim = KV_nope.shape
    head_dim_nope = BLOCK_DMODEL_NOPE
    head_dim_rope = BLOCK_DMODEL_ROPE

    # Compute strides
    stride_kv_batch, stride_kv_head, stride_kv_dmodel_nope = KV_nope.stride()
    _, _, stride_kv_dmodel_rope = KV_rope.stride()
    stride_o_batch, stride_o_head, stride_o_dmodel_nope = O_nope.stride()
    _, _, stride_o_dmodel_rope = O_rope.stride()

    # Launch kernel
    grid = (batch_size, num_heads)
    num_warps = 4
    num_stages = 2

    _fwd_kernel_destindex_copy_kv[grid](
        KV_nope, KV_rope, DestLoc, O_nope, O_rope,
        batch_size, num_heads, head_dim_nope, head_dim_rope,
        stride_kv_batch, stride_kv_head, stride_kv_dmodel_nope,
        stride_kv_dmodel_rope, stride_o_batch, stride_o_head,
        stride_o_dmodel_nope, stride_o_dmodel_rope,
        BLOCK_SIZE=BLOCK_DMODEL_NOPE,
        num_warps=num_warps,
        num_stages=num_stages
    )
