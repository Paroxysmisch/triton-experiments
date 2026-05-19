import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    KV_nope_ptr, KV_rope_ptr, DestLoc_ptr, O_nope_ptr, O_rope_ptr,
    batch_size, num_heads, head_dim,
    KV_nope_stride_b, KV_nope_stride_h, KV_nope_stride_d,
    KV_rope_stride_b, KV_rope_stride_h, KV_rope_stride_d,
    DestLoc_stride_b, DestLoc_stride_h,
    O_nope_stride_b, O_nope_stride_h, O_nope_stride_d,
    O_rope_stride_b, O_rope_stride_h, O_rope_stride_d,
    BLOCK_DMODEL_NOPE: tl.constexpr, BLOCK_DMODEL_ROPE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_heads_per_batch = num_heads
    num_blocks_per_head = (head_dim + BLOCK_DMODEL_NOPE - 1) // BLOCK_DMODEL_NOPE
    num_blocks_per_batch = num_heads_per_batch * num_blocks_per_head
    head_id = (pid % num_blocks_per_batch) // num_blocks_per_head
    block_id = (pid % num_blocks_per_batch) % num_blocks_per_head
    batch_id = pid // num_blocks_per_batch

    # Compute the offset for the current batch and head
    KV_nope_offset = batch_id * KV_nope_stride_b + head_id * KV_nope_stride_h
    KV_rope_offset = batch_id * KV_rope_stride_b + head_id * KV_rope_stride_h
    DestLoc_offset = batch_id * DestLoc_stride_b + head_id * DestLoc_stride_h
    O_nope_offset = batch_id * O_nope_stride_b + head_id * O_nope_stride_h
    O_rope_offset = batch_id * O_rope_stride_b + head_id * O_rope_stride_h

    # Compute the range of elements to process in this block
    block_start = block_id * BLOCK_DMODEL_NOPE
    block_end = min(block_start + BLOCK_DMODEL_NOPE, head_dim)

    # Load the destination indices for this block
    dest_indices = tl.load(DestLoc_ptr + DestLoc_offset + tl.arange(0, BLOCK_DMODEL_NOPE), mask=block_start + tl.arange(0, BLOCK_DMODEL_NOPE) < head_dim, other=0)

    # Load the source elements for this block
    src_nope = tl.load(KV_nope_ptr + KV_nope_offset + block_start, mask=block_start + tl.arange(0, BLOCK_DMODEL_NOPE) < head_dim, other=0)
    src_rope = tl.load(KV_rope_ptr + KV_rope_offset + block_start, mask=block_start + tl.arange(0, BLOCK_DMODEL_NOPE) < head_dim, other=0)

    # Store the elements in the destination tensors
    tl.store(O_nope_ptr + O_nope_offset + dest_indices, src_nope, mask=dest_indices < head_dim)
    tl.store(O_rope_ptr + O_rope_offset + dest_indices, src_rope, mask=dest_indices < head_dim)

import torch

def destindex_copy_kv(KV_nope, KV_rope, DestLoc, O_nope, O_rope, BLOCK_DMODEL_NOPE, BLOCK_DMODEL_ROPE):
    # Check that the shapes of corresponding source and destination tensors match
    assert KV_nope.shape == O_nope.shape, "Shapes of KV_nope and O_nope must match"
    assert KV_rope.shape == O_rope.shape, "Shapes of KV_rope and O_rope must match"
    assert DestLoc.shape == (KV_nope.shape[0], KV_nope.shape[1]), "Shape of DestLoc must match (batch_size, num_heads)"

    # Get the dimensions
    batch_size, num_heads, head_dim = KV_nope.shape

    # Prepare the aligned dimensions for kernel execution
    grid = (batch_size * num_heads * ((head_dim + BLOCK_DMODEL_NOPE - 1) // BLOCK_DMODEL_NOPE),)

    # Calculate strides
    KV_nope_stride_b, KV_nope_stride_h, KV_nope_stride_d = KV_nope.stride()
    KV_rope_stride_b, KV_rope_stride_h, KV_rope_stride_d = KV_rope.stride()
    DestLoc_stride_b, DestLoc_stride_h = DestLoc.stride()
    O_nope_stride_b, O_nope_stride_h, O_nope_stride_d = O_nope.stride()
    O_rope_stride_b, O_rope_stride_h, O_rope_stride_d = O_rope.stride()

    # Invoke the kernel
    _fwd_kernel_destindex_copy_kv[grid](
        KV_nope, KV_rope, DestLoc, O_nope, O_rope,
        batch_size, num_heads, head_dim,
        KV_nope_stride_b, KV_nope_stride_h, KV_nope_stride_d,
        KV_rope_stride_b, KV_rope_stride_h, KV_rope_stride_d,
        DestLoc_stride_b, DestLoc_stride_h,
        O_nope_stride_b, O_nope_stride_h, O_nope_stride_d,
        O_rope_stride_b, O_rope_stride_h, O_rope_stride_d,
        BLOCK_DMODEL_NOPE, BLOCK_DMODEL_ROPE,
        num_warps=4, num_stages=2
    )
