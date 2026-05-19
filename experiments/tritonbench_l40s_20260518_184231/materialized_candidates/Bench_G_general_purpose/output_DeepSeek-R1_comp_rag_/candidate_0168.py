import torch
import triton
import triton.language as tl

@triton.jit
def _copy_to_kcache_seqlen_n_kernel(
    K,
    KCache,
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
    cur_token_shift = cur_token_idx - (cur_seq_idx * n_tokens)  # Corrected calculation
    
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

def copy_k_to_blocked_cache(
    k: torch.Tensor,
    k_cache: torch.Tensor,
    kv_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    n: int = 1,
    use_new_kcache_layout: bool = False,
):
    if k.dim() == 4:
        k = k.reshape(-1, k.size(-2), k.size(-1))
    k_shape = k.shape
    bsz, num_kv_heads, head_dim = k_shape
    
    if n > 1:
        assert bsz % n == 0, "Batch size must be divisible by n_tokens"
        bsz = bsz // n

    assert kv_lengths.shape[0] == bsz and block_tables.shape[0] == bsz, "Dimension mismatch"

    block_size = k_cache.shape[-2] if not use_new_kcache_layout else k_cache.shape[3]
    x = head_dim
    stride_kcsplit_x, stride_kcs, stride_kcx = 0, k_cache.stride(2), k_cache.stride(3)
    
    if use_new_kcache_layout:
        x = k_cache.size(-1)
        stride_kcsplit_x, stride_kcs, stride_kcx = k_cache.stride(2), k_cache.stride(3), k_cache.stride(4)
        assert k_cache.shape[2] * x == head_dim, "Head dim mismatch"

    num_warps = 8 if head_dim > 128 else 4
    grid = (bsz * n, num_kv_heads, head_dim // x)
    
    _copy_to_kcache_seqlen_n_kernel[grid](
        K=k, KCache=k_cache, BLOCK_TABLES=block_tables, seq_lengths=kv_lengths,
        stride_kt=k.stride(0), stride_kh=k.stride(1), stride_kd=k.stride(2),
        stride_kcb=k_cache.stride(0), stride_kch=k_cache.stride(1),
        stride_kcsplit_x=stride_kcsplit_x, stride_kcs=stride_kcs,
        stride_kcx=stride_kcx, stride_bts=block_tables.stride(0),
        stride_btb=block_tables.stride(1), block_size=block_size,
        n_tokens=n, HEAD_DIM=head_dim, KCACHE_X=x, num_warps=num_warps
    )
