import torch
import triton
import triton.language as tl

# Triton 2.1.0
# supports two types of cache layouts
# 1. [num_blocks, num_kv_heads, block_size, head_dim]
# 2. [num_blocks, num_kv_heads, head_dim // x, block_size, x]
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
    # `n_tokens` is used to specify the number of tokens to copy for each sequence
    # When n_tokens > 1, tokens from different sequences are packed into the first dimension of the grid,
    #   `seq_lengths` must be the lengths of sequences counting the number of tokens to copy
    #   E.g. if n_tokens = 5, seq_lengths = [12, 15], then the already-copied position ids are [0-6, 0-9]
    #   for the two sequences, respectively. And the position ids to be copied are [7-11, 9-14].
    # When n_tokens = 1, consider token idx as the sequence idx, since it's only used during regular decoding stage
    cur_token_idx = tl.program_id(0)
    cur_seq_idx = cur_token_idx // n_tokens
    # `cur_token_shift` is only valid and functional when `n_tokens` > 1
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


# Triton 2.1.0
@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    K,
    V,
    KCache,
    VCache,
    BLOCK_TABLES,
    context_lengths,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_kcb,
    stride_kch,
    stride_kcsplit_x,
    stride_kcs,
    stride_kcd,
    stride_vcb,
    stride_vch,
    stride_vcs,
    stride_vcd,
    stride_bts,
    stride_btb,
    block_size,
    HEAD_DIM: tl.constexpr,
    KCACHE_X: tl.constexpr,
):
    cur_seq_idx = tl.program_id(0)
    cur_kv_head_idx = tl.program_id(1)

    past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - 1
    last_bt_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb)
    offsets_in_last_block = past_kv_seq_len % block_size

    range_x = tl.arange(0, KCACHE_X)
    offsets_dmodel_x_partition = tl.arange(0, KCACHE_X)

    for split_x in tl.static_range(HEAD_DIM // KCACHE_X):
        offsets_dmodel_x_partition = tl.arange(split_x * KCACHE_X, (split_x + 1) * KCACHE_X)
        offsets_k = cur_seq_idx * stride_kt + cur_kv_head_idx * stride_kh + offsets_dmodel_x_partition * stride_kd
        k = tl.load(K + offsets_k)
        offsets_v = cur_seq_idx * stride_vt + cur_kv_head_idx * stride_vh + offsets_dmodel_x_partition * stride_vd
        v = tl.load(V + offsets_v)

        offsets_kcache = (
            block_id * stride_kcb
            + cur_kv_head_idx * stride_kch
            + split_x * stride_kcsplit_x
            + offsets_in_last_block * stride_kcs
            + range_x
        )
        tl.store(KCache + offsets_kcache, k)
        offsets_vcache = (
            block_id * stride_vcb
            + cur_kv_head_idx * stride_vch
            + offsets_in_last_block * stride_vcs
            + offsets_dmodel_x_partition * stride_vcd
        )
        tl.store(VCache + offsets_vcache, v)
    return


def copy_k_to_blocked_cache(
    k: torch.Tensor,
    k_cache: torch.Tensor,
    kv_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    n: int = 1,
    use_new_kcache_layout: bool = False,
):
    """
    Copy keys or values to the blocked key/value cache during decoding stage.

    Args:
        k (torch.Tensor): [bsz, 1, num_kv_heads, head_dim]/[bsz, num_kv_heads, head_dim] - Keys or values during decoding with seq len 1.
            [bsz * n, num_kv_heads, head_dim] - Keys or values with seq len n
        k_cache (torch.Tensor): [num_blocks, num_kv_heads, block_size, head_dim] - Blocked key or value cache.
