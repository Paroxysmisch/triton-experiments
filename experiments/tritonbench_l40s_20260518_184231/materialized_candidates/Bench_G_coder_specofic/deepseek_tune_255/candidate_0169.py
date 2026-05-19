import torch
import triton
import triton.language as tl

# Triton kernel for copying key/value vectors to blocked cache
@triton.jit
def _copy_to_kcache_seqlen_n_kernel(
    kv_ptr, kv_stride_b, kv_stride_h, kv_stride_d,
    kcache_ptr, kcache_stride_bl, kcache_stride_sl, kcache_stride_h, kcache_stride_d,
    cur_token_indices_ptr, cur_seq_index_ptr,
    past_seq_lens_ptr, past_seq_lens_stride_b,
    kcache_x, kcache_y,
    batch_size, nheads, kv_dim, seqlen,
    CACHE_FORMAT_V2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    cur_token_index = tl.program_id(0)
    cur_seq_index = tl.load(cur_seq_index_ptr)
    cur_batch_index = tl.program_id(1)
    cur_head_index = tl.program_id(2)

    past_seq_len = 0 if past_seq_lens_ptr is None else tl.load(past_seq_lens_ptr + cur_batch_index * past_seq_lens_stride_b)
    cur_seq_index = cur_seq_index + past_seq_len

    if cur_seq_index * nheads * BLOCK_DIM + cur_token_index >= cur_token_indices_ptr[cur_batch_index]:
        return

    cur_kv_offset = cur_head_index * kv_stride_h + cur_token_index * kv_stride_d
    kv_ptrs = kv_ptr + cur_batch_index * kv_stride_b + cur_kv_offset
    kv = tl.load(kv_ptrs, mask=cur_head_index < nheads, other=0.0)

    cur_kcache_offset = cur_head_index * kcache_stride_h + cur_token_index * kcache_stride_d
    kcache_ptrs = kcache_ptr + cur_batch_index * kcache_stride_bl + cur_seq_index * kcache_stride_sl + cur_kcache_offset
    tl.store(kcache_ptrs, kv, mask=cur_head_index < nheads)

# Function to call the Triton kernel
def copy_k_to_blocked_cache(k, kcache, cur_token_indices, cur_seq_index, past_seq_lens=None, kcache_format_v2=False):
    assert k.ndim == 3
    assert k.shape == kcache.shape
    batch_size, nheads, kv_dim = k.shape
    kv_stride_b, kv_stride_h, kv_stride_d = k.stride()
    kcache_stride_bl, kcache_stride_sl, kcache_stride_h, kcache_stride_d, kcache_stride_k = kcache.stride()

    if kcache_format_v2:
        assert kcache_stride_k == kcache_stride_d
        kcache_x = kcache_stride_sl
        kcache_y = kcache_stride_k
    else:
        kcache_x = kcache_stride_sl
        kcache_y = kcache_stride_d

    kcache = kcache.contiguous()
    if kcache_format_v2:
        kcache = kcache.reshape((batch_size, kcache_x, kcache_y * nheads))
    else:
        kcache = kcache.reshape((batch_size * kcache_x, nheads, kcache_y))

    HEAD_DIM = kv_stride_d
    BLOCK_DIM = get_block_dim(kv_dim)
    grid = (triton.cdiv(kv_dim, BLOCK_DIM), cur_seq_index, batch_size, nheads)

    _copy_to_kcache_seqlen_n_kernel[grid](
        k, kv_stride_b, kv_stride_h, kv_stride_d,
        kcache, kcache_stride_bl, kcache_stride_sl, kcache_stride_h, kcache_stride_d,
        cur_token_indices, cur_seq_index,
        past_seq_lens, kcache_stride_bl if past_seq_lens is not None else 0,
        kcache_x, kcache_y,
        batch_size, nheads, kv_dim, seqlen,
        kcache_format_v2,
        HEAD_DIM=HEAD_DIM,
        BLOCK_DIM=BLOCK_DIM,
        num_warps=1,
    )
