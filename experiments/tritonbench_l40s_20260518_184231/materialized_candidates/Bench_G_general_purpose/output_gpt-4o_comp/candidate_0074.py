import triton
import triton.language as tl

@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    K_ptr, V_ptr, KCache_ptr, VCache_ptr, block_tables_ptr, context_lengths_ptr,
    stride_kb, stride_kh, stride_kd, stride_vb, stride_vh, stride_vd,
    stride_kcb, stride_kch, stride_kcd1, stride_kcd2, stride_vcb, stride_vch, stride_vcd1, stride_vcd2,
    num_heads, head_dim, block_size, use_new_cache_layout: tl.constexpr
):
    # Program ID
    seq_idx = tl.program_id(0)  # sequence index
    head_idx = tl.program_id(1)  # head index

    # Offsets for input K and V tensors
    k_offset = seq_idx * stride_kb + head_idx * stride_kh
    v_offset = seq_idx * stride_vb + head_idx * stride_vh

    # Load keys and values for this sequence and head
    k = tl.load(K_ptr + k_offset + tl.arange(0, head_dim) * stride_kd)
    v = tl.load(V_ptr + v_offset + tl.arange(0, head_dim) * stride_vd)

    # Retrieve block index and past context length
    block_idx = tl.load(block_tables_ptr + seq_idx)
    context_length = tl.load(context_lengths_ptr + seq_idx)

    # Calculate offsets for cache tensors
    if use_new_cache_layout:
        kc_offset = (
            block_idx * stride_kcb + head_idx * stride_kch +
            (tl.arange(0, head_dim) // block_size) * stride_kcd1 +
            context_length * stride_kcd2 +
            (tl.arange(0, head_dim) % block_size)
        )
        vc_offset = (
            block_idx * stride_vcb + head_idx * stride_vch +
            (tl.arange(0, head_dim) // block_size) * stride_vcd1 +
            context_length * stride_vcd2 +
            (tl.arange(0, head_dim) % block_size)
        )
    else:
        kc_offset = (
            block_idx * stride_kcb + head_idx * stride_kch +
            context_length * stride_kcd1 +
            tl.arange(0, head_dim) * stride_kcd2
        )
        vc_offset = (
            block_idx * stride_vcb + head_idx * stride_vch +
            context_length * stride_vcd1 +
            tl.arange(0, head_dim) * stride_vcd2
        )

    # Store keys and values into the cache
    tl.store(KCache_ptr + kc_offset, k)
    tl.store(VCache_ptr + vc_offset, v)

import torch

def copy_kv_to_blocked_cache(
    K, V, KCache, VCache, block_tables, context_lengths, block_size, use_new_cache_layout=False
):
    """
    Wrapper function to copy K and V tensors to their respective cache tensors.

    Args:
        K: [batch_size, num_heads, head_dim] Keys tensor.
        V: [batch_size, num_heads, head_dim] Values tensor.
        KCache: Cache tensor for keys.
        VCache: Cache tensor for values.
        block_tables: Mapping of blocks for each sequence.
        context_lengths: Lengths of past sequences.
        block_size: Size of each block in the cache.
        use_new_cache_layout: Whether to use the new cache layout.
    """
    batch_size, num_heads, head_dim = K.shape
    num_blocks = KCache.shape[0]  # Assumes KCache has shape [num_blocks, ...]

    # Ensure compatibility of shapes
    assert V.shape == (batch_size, num_heads, head_dim), "K and V must have the same shape"
    assert block_tables.shape[0] == batch_size, "Block tables must match batch size"
    assert context_lengths.shape[0] == batch_size, "Context lengths must match batch size"

    # Calculate strides for input tensors
    stride_kb, stride_kh, stride_kd = K.stride()
    stride_vb, stride_vh, stride_vd = V.stride()

    # Calculate strides for cache tensors
    if use_new_cache_layout:
        stride_kcb, stride_kch, stride_kcd1, stride_kcd2, _ = KCache.stride()
        stride_vcb, stride_vch, stride_vcd1, stride_vcd2, _ = VCache.stride()
    else:
        stride_kcb, stride_kch, stride_kcd1, stride_kcd2 = KCache.stride()
        stride_vcb, stride_vch, stride_vcd1, stride_vcd2 = VCache.stride()

    # Launch the Triton kernel
    grid = (batch_size, num_heads)  # One sequence and head per program
    _copy_to_kvcache_seqlen1_kernel[grid](
        K, V, KCache, VCache, block_tables, context_lengths,
        stride_kb, stride_kh, stride_kd, stride_vb, stride_vh, stride_vd,
        stride_kcb, stride_kch, stride_kcd1, stride_kcd2,
        stride_vcb, stride_vch, stride_vcd1, stride_vcd2,
        num_heads, head_dim, block_size, use_new_cache_layout
    )
