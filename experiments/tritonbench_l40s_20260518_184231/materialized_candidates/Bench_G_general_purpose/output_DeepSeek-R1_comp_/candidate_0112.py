import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv(
    # Pointers to tensors
    KV_nope_ptr,
    KV_rope_ptr,
    DestLoc_ptr,
    O_nope_ptr,
    O_rope_ptr,
    # Tensor dimensions
    batch_size,
    num_heads,
    d_model_nope,
    d_model_rope,
    # Stride information
    kvn_batch_stride,
    kvn_head_stride,
    kvn_dmodel_stride,
    kvr_batch_stride,
    kvr_head_stride,
    kvr_dmodel_stride,
    destloc_batch_stride,
    destloc_head_stride,
    on_batch_stride,
    on_head_stride,
    on_seq_stride,
    on_dmodel_stride,
    or_batch_stride,
    or_head_stride,
    or_seq_stride,
    or_dmodel_stride,
    # Block sizes
    BLOCK_DMODEL_NOPE: tl.constexpr,
    BLOCK_DMODEL_ROPE: tl.constexpr,
    # Number of blocks
    n_nope_blocks: tl.constexpr,
    n_rope_blocks: tl.constexpr,
):
    # Get current program IDs
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_block = tl.program_id(2)

    # Calculate destination index
    dest_offset = pid_batch * destloc_batch_stride + pid_head * destloc_head_stride
    dest_idx = tl.load(DestLoc_ptr + dest_offset)

    # Process either nope or rope based on block index
    if pid_block < n_nope_blocks:
        # Process nope component
        block_start = pid_block * BLOCK_DMODEL_NOPE
        offsets = block_start + tl.arange(0, BLOCK_DMODEL_NOPE)
        mask = offsets < d_model_nope

        # Load from KV_nope
        kv_ptr = (KV_nope_ptr + 
                 pid_batch * kvn_batch_stride + 
                 pid_head * kvn_head_stride + 
                 offsets * kvn_dmodel_stride)
        data = tl.load(kv_ptr, mask=mask)

        # Store to O_nope
        o_ptr = (O_nope_ptr + 
                pid_batch * on_batch_stride + 
                pid_head * on_head_stride + 
                dest_idx * on_seq_stride + 
                offsets * on_dmodel_stride)
        tl.store(o_ptr, data, mask=mask)
    else:
        # Process rope component
        rope_block_idx = pid_block - n_nope_blocks
        block_start = rope_block_idx * BLOCK_DMODEL_ROPE
        offsets = block_start + tl.arange(0, BLOCK_DMODEL_ROPE)
        mask = offsets < d_model_rope

        # Load from KV_rope
        kv_ptr = (KV_rope_ptr + 
                 pid_batch * kvr_batch_stride + 
                 pid_head * kvr_head_stride + 
                 offsets * kvr_dmodel_stride)
        data = tl.load(kv_ptr, mask=mask)

        # Store to O_rope
        o_ptr = (O_rope_ptr + 
                pid_batch * or_batch_stride + 
                pid_head * or_head_stride + 
                dest_idx * or_seq_stride + 
                offsets * or_dmodel_stride)
        tl.store(o_ptr, data, mask=mask)

def destindex_copy_kv(KV_nope: torch.Tensor, 
                     KV_rope: torch.Tensor, 
                     DestLoc: torch.Tensor, 
                     O_nope: torch.Tensor, 
                     O_rope: torch.Tensor):
    # Shape validation
    assert KV_nope.shape[:2] == O_nope.shape[:2], "Nope shape mismatch"
    assert KV_rope.shape[:2] == O_rope.shape[:2], "Rope shape mismatch"
    assert KV_nope.shape[:2] == DestLoc.shape, "DestLoc shape mismatch"

    batch_size, num_heads = KV_nope.shape[:2]
    d_model_nope = KV_nope.size(2)
    d_model_rope = KV_rope.size(2)

    # Calculate block sizes as next power of two
    BLOCK_DMODEL_NOPE = triton.next_power_of_2(d_model_nope)
    BLOCK_DMODEL_ROPE = triton.next_power_of_2(d_model_rope)

    # Calculate number of blocks needed for each dimension
    n_nope_blocks = (d_model_nope + BLOCK_DMODEL_NOPE - 1) // BLOCK_DMODEL_NOPE
    n_rope_blocks = (d_model_rope + BLOCK_DMODEL_ROPE - 1) // BLOCK_DMODEL_ROPE
    total_blocks = n_nope_blocks + n_rope_blocks

    # Grid configuration
    grid = (batch_size, num_heads, total_blocks)

    # Get tensor strides
    kvn_strides = KV_nope.stride()
    kvr_strides = KV_rope.stride()
    dest_strides = DestLoc.stride()
    on_strides = O_nope.stride()
    or_strides = O_rope.stride()

    # Launch kernel
    _fwd_kernel_destindex_copy_kv[grid](
        KV_nope, KV_rope, DestLoc, O_nope, O_rope,
        batch_size, num_heads, d_model_nope, d_model_rope,
        kvn_strides[0], kvn_strides[1], kvn_strides[2],
        kvr_strides[0], kvr_strides[1], kvr_strides[2],
        dest_strides[0], dest_strides[1],
        on_strides[0], on_strides[1], on_strides[2], on_strides[3],
        or_strides[0], or_strides[1], or_strides[2], or_strides[3],
        BLOCK_DMODEL_NOPE, BLOCK_DMODEL_ROPE,
        n_nope_blocks, n_rope_blocks,
        num_warps=4,
        num_stages=4
    )
