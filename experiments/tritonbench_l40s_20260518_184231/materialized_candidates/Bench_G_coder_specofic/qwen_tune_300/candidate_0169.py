import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def _copy_to_kcache_seqlen_n_kernel(
    k,
    k_cache,
    kv_index,
    past_seq_len,
    seq_len,
    abs_past_seq_len,
    HEAD_DIM: tl.constexpr,
    KCACHE_X: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Convert kernel arguments to triton-compatible types
    k = k.to(tl.pointer_type(tl.float16))
    k_cache = k_cache.to(tl.pointer_type(tl.float16))
    kv_index = kv_index.to(tl.pointer_type(tl.int32))
    past_seq_len = past_seq_len.to(tl.pointer_type(tl.int32))
    seq_len = seq_len.to(tl.pointer_type(tl.int32))
    abs_past_seq_len = abs_past_seq_len.to(tl.pointer_type(tl.int32))

    # Load sequence lengths and indices
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_token = tl.program_id(2)
    cur_kv = tl.program_id(3)

    kv_index += cur_batch * 2 + cur_kv
    kv_index = tl.load(kv_index)

    seq_len += cur_batch
    seq_len = tl.load(seq_len)

    past_seq_len += cur_batch
    past_seq_len = tl.load(past_seq_len)

    abs_past_seq_len += cur_batch
    abs_past_seq_len = tl.load(abs_past_seq_len)

    cur_token += abs_past_seq_len

    # Determine if the current thread is responsible for copying data
    if cur_token < seq_len:
        cur_token_offset = cur_token.to(tl.int64) * HEAD_DIM + tl.arange(0, BLOCK)
        cur_token_offset = tl.broadcast_to(cur_token_offset, (BLOCK, HEAD_DIM))

        kv_index_offset = kv_index * KCACHE_X + tl.arange(0, HEAD_DIM) // (HEAD_DIM // KCACHE_X)
        kv_index_offset = tl.broadcast_to(kv_index_offset, (BLOCK, HEAD_DIM))

        # Load data from k and store it into k_cache
        k_ptr = k + cur_token_offset
        k_cache_ptr = k_cache + kv_index_offset
        k_val = tl.load(k_ptr, mask=cur_token_offset < seq_len * HEAD_DIM, other=0.0)
        tl.store(k_cache_ptr, k_val, mask=cur_token_offset < seq_len * HEAD_DIM)
    return

@torch.no_grad()
def copy_k_to_blocked_cache(
    k: Tensor,
    k_cache: Tensor,
    kv_index: Tensor,
    past_seq_len: Tensor,
    abs_past_seq_len: Tensor,
) -> None:
    """
    Copy key or value vectors to the blocked cache.
    Args:
        k: (batch, seq_len, head_dim)
        k_cache: (batch, num_kv, head_dim // k, seq_len, k)
        kv_index: (batch, 2)
        past_seq_len: (batch,)
        abs_past_seq_len: (batch,)
    Returns:
        None
    """
    assert k.shape[-1] == k_cache.shape[-2]
    assert k.is_contiguous()

    if k_cache.stride(-1) != 1:
        k_cache = k_cache.contiguous()

    HEAD_DIM = k.shape[-1]
    KCACHE_X = k_cache.shape[-2]
    BLOCK = HEAD_DIM // KCACHE_X

    k = k.view((k.shape[0], k.shape[1], BLOCK, -1)).transpose(2, 3).reshape(-1, BLOCK, HEAD_DIM)
    seq_len = k.shape[1]

    grid = (triton.cdiv(seq_len, k_cache.shape[-1]), k_cache.shape[1], k_cache.shape[2], 2)

    _copy_to_kcache_seqlen_n_kernel[grid](
        k,
        k_cache,
        kv_index,
        past_seq_len,
        seq_len,
        abs_past_seq_len,
        HEAD_DIM,
        KCACHE_X,
        BLOCK,
    )
    return
