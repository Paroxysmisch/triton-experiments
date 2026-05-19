import torch
import triton
import triton.language as tl

@triton.jit
def _copy_to_kcache_seqlen_n_kernel(
    K,  # K or V
    KCache,  # [num_blocks, num_kv_heads, head_dim // x, block_size, x]
    BLOCK_TABLES,
    seq_lengths,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_kcb,
    stride_kch,
    stride_kcsplit_x,
    stride_kcs,
    stride_kcx,
    stride_bts,
    stride_btb,
    block_size,
    n_tokens,
    HEAD_DIM: tl.constexpr,
    KCACHE_X: tl.constexpr,
):
    cur_token_idx = tl.program_id(0)
    cur_seq_idx = cur_token_idx // n_tokens
    cur_token_shift = cur_token_idx - (n_tokens * (cur_seq_idx + 1))
    cur_kv_head_idx = tl.program_id(1)
    split_x_idx = tl.program_id(2)

    past_kv_seq_len = tl.load(seq_lengths + cur_seq_idx) + cur_token_shift
    last_bt_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb)
    offset_last_block = past_kv_seq_len % block_size
    offsets_dmodel = split_x_idx * KCACHE_X + tl.arange(0, KCACHE_X)
    offsets_k = cur_token_idx * stride_kt + cur_kv_head_idx * stride_kh + offsets_dmodel * stride_kd
    k = tl.load(K + offsets_k)
    offsets_kcache = (
        block_id * stride_kcb
        + cur_kv_head_idx * stride_kch
        + split_x_idx * stride_kcsplit_x
        + offset_last_block * stride_kcs
        + tl.arange(0, KCACHE_X)
    )
    tl.store(KCache + offsets_kcache, k)
    return

def copy_k_to_blocked_cache(
    k: torch.Tensor,
    k_cache: torch.Tensor,
    kv_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    n: int = 1,
    use_new_kcache_layout: bool = False,
):
    assert k.dtype == k_cache.dtype, "Expected consistent dtype for tensor and cache."
    if k.dim() == 4:
        k = k.reshape(-1, k.size(-2), k.size(-1))
    k_shape = k.shape
    bsz, num_kv_heads, head_dim = k_shape
    if n > 1:
        assert bsz % n == 0, "Each sequence should have the same number of tokens to be copied"
        bsz = bsz // n

    assert kv_lengths.shape[0] == block_tables.shape[0] == bsz, (
        f"Got incompatible batch size (number of seqs):\n"
        f"  Past kv sequence lengths bsz {kv_lengths.shape[0]}; "
        f" block tables bsz {block_tables.shape[0]}, input k batch size {bsz}"
    )

    k_cache_shape = k_cache.shape
    block_size = k_cache_shape[-2]

    x = head_dim
    stride_kcsplit_x, stride_kcs, stride_kcd = 0, k_cache.stride(2), k_cache.stride(3)
    if use_new_kcache_layout:
        assert (
            len(k_cache_shape) == 5
            and k_cache_shape[1] == k_shape[1]
            and k_cache_shape[2] * k_cache_shape[4] == k_shape[2]
        ), f"Incompatible k_cache shape {k_cache_shape} with k shape {k_shape}"
        x = k_cache.size(-1)
        stride_kcsplit_x, stride_kcs, stride_kcd = k_cache.stride()[2:]

    num_warps = 8 if head_dim > 128 else 4
    grid = (bsz * n, num_kv_heads, head_dim // x)
    _copy_to_kcache_seqlen_n_kernel[grid](
        k,
        k_cache,
        block_tables,
        kv_lengths,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        stride_kcsplit_x,
        stride_kcs,
        stride_kcd,
        block_tables.stride(0),
        block_tables.stride(1),
        block_size,
        n_tokens=n,
        HEAD_DIM=head_dim,
        KCACHE_X=x,
        num_warps=num_warps,
    )
