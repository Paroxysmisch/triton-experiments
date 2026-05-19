import torch
import triton
import triton.language as tl

@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    # Input tensors
    K,  # [batch_size, num_kv_heads, head_dim]
    V,  # [batch_size, num_kv_heads, head_dim]
    # The following four tensors are all
    # [batch_size, num_kv_heads, block_size, head_dim]
    KCache,
    VCache,
    # Block index table
    # [num_seqs, num_kv_heads, max_num_blocks_per_seq]
    block_tables,
    # The following two tensors are both
    # [num_seqs, max_num_blocks_per_seq]
    context_lengths,
    seq_idx_in_block_table,
    # Meta-params
    stride_k_bs: int,
    stride_k_h: int,
    stride_k_d: int,
    stride_v_bs: int,
    stride_v_h: int,
    stride_v_d: int,
    stride_kcache_bs: int,
    stride_kcache_h: int,
    stride_kcache_d: int,
    stride_kcache_bl: int,
    stride_vcache_bs: int,
    stride_vcache_h: int,
    stride_vcache_d: int,
    stride_vcache_bl: int,
    BLOCK: tl.constexpr,
    HEAD_DIM_K: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    USE_TENSOR_CORES: tl.constexpr = False,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    tl.static_assert(HEAD_DIM_K == HEAD_DIM_V, "Head dim of K and V must be the same")
    cur_seq_in_block_table = tl.load(seq_idx_in_block_table + cur_seq)
    cur_block = tl.load(
        block_tables + cur_seq_in_block_table * BLOCK * BLOCK + cur_head * BLOCK * BLOCK
    )
    offs_d = tl.arange(0, HEAD_DIM_K)
    offs_block = tl.arange(0, BLOCK)
    cur_token = context_lengths + cur_seq * BLOCK * BLOCK + cur_block * BLOCK * BLOCK
    cur_token += offs_block[:, None] * BLOCK

    # Load K, V
    k_ptrs = K + cur_seq * stride_k_bs + cur_head * stride_k_h + offs_d[None, :]
    v_ptrs = V + cur_seq * stride_v_bs + cur_head * stride_v_h + offs_d[None, :]
    k = tl.load(
        k_ptrs,
        mask=cur_token[:, None] < context_lengths + cur_seq * BLOCK * BLOCK,
        other=0.0,
    )
    v = tl.load(
        v_ptrs,
        mask=cur_token[:, None] < context_lengths + cur_seq * BLOCK * BLOCK,
        other=0.0,
    )

    # Write K, V to KCache, VCache
    k_ptrs = (
        KCache
        + cur_seq * stride_kcache_bs
        + cur_head * stride_kcache_h
        + offs_block[:, None] * stride_kcache_bl
        + offs_d[None, :]
    )
    v_ptrs = (
        VCache
        + cur_seq * stride_vcache_bs
        + cur_head * stride_vcache_h
        + offs_block[:, None] * stride_vcache_bl
        + offs_d[None, :]
    )
    tl.store(k_ptrs, k, mask=cur_token[:, None] < cur_seq * BLOCK * BLOCK)
    tl.store(v_ptrs, v, mask=cur_token[:, None] < cur_seq * BLOCK * BLOCK)


def copy_kv_to_blocked_cache(
    kv_cache_dtype: str,
    x: int,
    K: torch.Tensor,
    V: torch.Tensor,
    KCache: torch.Tensor,
    VCache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    seq_idx_in_block_table: torch.Tensor,
):
    assert K.shape[-1] == V.shape[-1]
    assert K.shape[-1] in {16, 32, 64, 128, 256}
    head_dim = K.shape[-1]
    # Grid size is (batch_size, num_heads)
    batch_size, num_heads, num_kv_heads = (
        context_lengths.shape[0],
        K.shape[1],
        V.shape[1],
    )
    grid = (batch_size, num_heads)
    num_blocks = block_tables.shape[-1]

    if kv_cache_dtype in ["float16", "bfloat16"]:
        if x == 1:
            _copy_to_kvcache_seqlen1_kernel[grid](
                K,
                V,
                KCache,
                VCache,
                block_tables,
                context_lengths,
                seq_idx_in_block_table,
                # strides of K
                stride_k_bs=K.stride(0),
                stride_k_h=K.stride(1),
                stride_k_d=K.stride(2),
                # strides of V
                stride_v_bs=V.stride(0),
                stride_v_h=V.stride(1),
                stride_v_d=V.stride(2),
                # strides of KCache
                stride_kcache_bs=KCache.stride(0),
                stride_kcache_h=KCache.stride(1),
                stride_kcache_d=KCache.stride(3),
                stride_kcache_bl=KCache.stride(2),
                # strides of VCache
                stride_vcache_bs=VCache.stride(0),
                stride_vcache_h=VCache.stride(1),
                stride_vcache_d=VCache.stride(3),
                stride_vcache_bl=VCache.stride(2),
                # meta-params
                BLOCK=num_blocks,
                HEAD_DIM_K=head_dim,
                HEAD_DIM_V=head_dim,
                num_warps=8,
                num_stages=2,
            )
        else:
            _copy_to_kvcache_seqlen1_kernel[grid](
                K,
                V,
                KCache,
                VCache,
                block_tables,
                context_lengths,
                seq_idx_in_block_table,
                # strides of K
                stride_k_bs=K.stride(0),
                stride_k_h=K.stride(1),
                stride_k_d=K.stride(2),
                # strides of V
                stride_v_bs=V.stride(0),
                stride_v_h=V.stride(1),
                stride_v_d=V.stride(2),
                # strides of KCache
                stride_kcache_bs=KCache.stride(0),
                stride_kcache_h=KCache.stride(1),
                stride_kcache_d=KCache.stride(2),
                stride_kcache_bl=KCache.stride(3),
                # strides of VCache
                stride_vcache_bs=VCache.stride(0),
                stride_vcache_h=VCache.stride(1),
                stride_vcache_d=VCache.stride(2),
                stride_vcache_bl=VCache.stride(3),
                # meta-params
                BLOCK=num_blocks,
                HEAD_DIM_K=head_dim // x,
                HEAD_DIM_V=head_dim // x,
                num_warps=8,
                num_stages=2,
            )
    else:
        if x == 1:
            _copy_to_kvcache_seqlen1_kernel[grid](
                K,
                V,
                KCache,
                VCache,
                block_tables,
                context_lengths,
                seq_idx_in_block_table,
                # strides of K
                stride_k_bs=K.stride(0),
                stride_k_h=K.stride(1),
                stride_k_d=K.stride(2),
                # strides of V
                stride_v_bs=V.stride(0),
                stride_v_h=V.stride(1),
                stride_v_d=V.stride(2),
                # strides of KCache
                stride_kcache_bs=KCache.stride(0),
                stride_kcache_h=KCache.stride(1),
                stride_kcache_d=KCache.stride(2),
                stride_kcache_bl=KCache.stride(3),
                # strides of VCache
                stride_vcache_bs=VCache.stride(0),
                stride_vcache_h=VCache.stride(1),
                stride_vcache_d=VCache.stride(2),
                stride_vcache_bl=VCache.stride(3),
                # meta-params
                BLOCK=num_blocks,
                HEAD_DIM_K=head_dim,
                HEAD_DIM_V=head_dim,
                USE_TENSOR_CORES=True,
                num_warps=8,
                num_stages=2,
            )
        else:
            _copy_to_kvcache_seqlen1_kernel[grid](
                K,
                V,
                KCache,
                VCache,
                block_tables,
                context_lengths,
                seq_idx_in_block_table,
                # strides of K
                stride_k_bs=K.stride(0),
                stride_k_h=K.stride(1),
                stride_k_d=K.stride(2),
                # strides of V
                stride_v_bs=V.stride(0),
                stride_v_h=V.stride(1),
                stride_v_d=V.stride(2),
                # strides of KCache
                stride_kcache_bs=KCache.stride(0),
                stride_kcache_h=KCache.stride(1),
                stride_kcache_d=KCache.stride(3),
                stride_kcache_bl=KCache.stride(2),
                # strides of VCache
                stride_vcache_bs=VCache.stride(0),
                stride_vcache_h=VCache.stride(1),
                stride_vcache_d=VCache.stride(3),
                stride_vcache_bl=VCache.stride(2),
                # meta-params
                BLOCK=num_blocks,
                HEAD_DIM_K=head_dim // x,
                HEAD
