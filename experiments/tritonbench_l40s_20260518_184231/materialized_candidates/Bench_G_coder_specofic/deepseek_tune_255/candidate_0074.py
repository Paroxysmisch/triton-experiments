import torch
import triton
import triton.language as tl

@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    K, V, KCache, VCache, block_tables, context_lengths,
    batch_size, num_blocks, num_heads, num_kv_heads,
    head_dim, block_size, x,
    stride_k_batch, stride_k_seqlen, stride_k_head, stride_k_dim,
    stride_v_batch, stride_v_seqlen, stride_v_head, stride_v_dim,
    stride_block_tables_batch, stride_block_tables_seqlen, stride_block_tables_head,
    stride_block_tables_dim,
    stride_cache_kv_batch, stride_cache_kv_blk, stride_cache_kv_head, stride_cache_kv_size, stride_cache_kv_dim,
    stride_cache_cl_batch, stride_cache_cl_seqlen,
    IS_TRITON_22: tl.constexpr, IS_TRITON_23: tl.constexpr,
):
    seq_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    num_heads_per_block = tl.cdiv(num_heads, num_kv_heads)
    kv_head_id = head_id // num_heads_per_block
    head_offset = head_id % num_heads_per_block

    if IS_TRITON_22:
        if IS_TRITON_23:
            block_id = tl.load(block_tables + seq_id * stride_block_tables_batch + head_offset * stride_block_tables_head)
        else:
            block_id = tl.load(block_tables + seq_id * stride_block_tables_batch + head_offset * stride_block_tables_head)
    else:
        block_id = tl.load(block_tables + seq_id * stride_block_tables_batch + head_offset * stride_block_tables_head)

    offs_k = seq_id * stride_k_batch + head_offset * stride_k_head
    offs_v = seq_id * stride_v_batch + head_offset * stride_v_head

    if block_id != -1:
        if x == 1:
            if IS_TRITON_22:
                if IS_TRITON_23:
                    tl.store(KCache + block_id * stride_cache_kv_blk + head_offset * stride_cache_kv_head,
                             kv_head_id * stride_cache_kv_head,
                             block_size * stride_cache_kv_size)
                    tl.store(VCache + block_id * stride_cache_kv_blk + head_offset * stride_cache_kv_head,
                             kv_head_id * stride_cache_kv_head,
                             block_size * stride_cache_kv_size)
                else:
                    tl.store(KCache + block_id * stride_cache_kv_blk + head_offset * stride_cache_kv_head,
                             kv_head_id * stride_cache_kv_head,
                             block_size * stride_cache_kv_size)
                    tl.store(VCache + block_id * stride_cache_kv_blk + head_offset * stride_cache_kv_head,
                             kv_head_id * stride_cache_kv_head,
                             block_size * stride_cache_kv_size)
            else:
                tl.store(KCache + block_id * stride_cache_kv_blk + head_offset * stride_cache_kv_head,
                         kv_head_id * stride_cache_kv_head,
                         block_size * stride_cache_kv_size)
                tl.store(VCache + block_id * stride_cache_kv_blk + head_offset * stride_cache_kv_head,
                         kv_head_id * stride_cache_kv_head,
                         block_size * stride_cache_kv_size)

        for i in range(0, block_size):
            k_ptrs = K + offs_k + i * stride_k_dim
            v_ptrs = V + offs_v + i * stride_v_dim
            if x == 1:
                if IS_TRITON_22:
                    if IS_TRITON_23:
                        k_cache_ptrs = KCache + block_id * stride_cache_kv_blk + head_offset * stride_cache_kv_head + i * stride_cache_kv_size
                        v_cache_ptrs = VCache + block_id * stride_cache_kv_blk + head_offset * stride_cache_kv_head + i * stride_cache_kv_size
                    else:
                        k_cache_ptrs = KCache + block_id * stride_cache_kv_blk + head_offset * stride_cache_kv_head + i * stride_cache_kv_size
                        v_cache_ptrs = VCache + block_id * stride_cache_kv_blk + head_offset * stride_cache_kv_head + i * stride_cache_kv_size
                else:
                    k_cache_ptrs = KCache + block_id * stride_cache_kv_blk + head_offset * stride_cache_kv_head + i * stride_cache_kv_size
                    v_cache_ptrs = VCache + block_id * stride_cache_kv_blk + head_offset * stride_cache_kv_head + i * stride_cache_kv_size

                tl.store(k_cache_ptrs, k_ptrs, head_dim, mask=i < block_size)
                tl.store(v_cache_ptrs, v_ptrs, head_dim, mask=i < block_size)


def copy_kv_to_blocked_cache(
    K: torch.Tensor, V: torch.Tensor, KCache: torch.Tensor, VCache: torch.Tensor,
    block_tables: torch.Tensor, context_lengths: torch.Tensor,
    new_cache_layout: bool = False,
):
    assert K.shape == V.shape
    assert K.ndim == V.ndim == 4
    assert K.size(1) == V.size(1)
    assert K.size(2) == V.size(2)
    assert KCache.shape == VCache.shape
    assert KCache.ndim == VCache.ndim == 5
    assert block_tables.ndim == 3
    assert context_lengths.ndim == 2
    assert context_lengths.size(1) == 1
    assert K.stride(1) == V.stride(1) == KCache.stride(1) == VCache.stride(1)
    assert K.stride(2) == V.stride(2) == 1
    assert KCache.stride(2) == VCache.stride(2) == 1
    assert block_tables.size(0) == K.size(0) == V.size(0)
    assert block_tables.size(2) == KCache.size(1) == VCache.size(1)
    assert context_lengths.size(0) == K.size(0)
    assert context_lengths.stride(0) == context_lengths.stride(1) == 1

    batch_size, num_blocks, num_heads, head_dim = K.shape
    num_kv_heads = KCache.shape[2] if new_cache_layout else 1
    block_size = KCache.
