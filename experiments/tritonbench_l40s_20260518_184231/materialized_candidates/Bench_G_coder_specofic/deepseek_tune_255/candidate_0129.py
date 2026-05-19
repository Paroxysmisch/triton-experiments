import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_apply_penalty(
    cur_batch,
    p_coef_pre,
    p_coef_freq,
    p_coef_rep,
    p_token_ids,
    p_token_counts,
    p_cumsum_seq_len,
    Logits,
    stride_token_ids,
    stride_token_counts,
    stride_logits_batch,
    stride_logits_seq,
    stride_logits_head,
    stride_logits_hidden,
    stride_logits_pre_rep,
    stride_logits_freq,
    BLOCK_TOKEN: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    coef_pre = tl.load(p_coef_pre + cur_batch)
    coef_freq = tl.load(p_coef_freq + cur_batch)
    coef_rep = tl.load(p_coef_rep + cur_batch)
    cumsum_seq_len = tl.load(p_cumsum_seq_len + cur_batch)
    cur_seq_len = tl.where(cur_batch == 0, 0, cumsum_seq_len - tl.load(p_cumsum_seq_len + cur_batch - 1))

    start_token_idx = tl.load(p_cumsum_seq_len + cur_batch - 1) if cur_batch > 0 else 0
    end_token_idx = start_token_idx + cur_seq_len

    range_token_idx = tl.arange(0, BLOCK_TOKEN) + start_token_idx
    mask_token = range_token_idx < end_token_idx

    range_batch_idx = tl.arange(0, BLOCK_BATCH) + cur_batch
    range_seq_idx = tl.arange(0, BLOCK_SEQ)
    range_head_idx = tl.arange(0, BLOCK_HIDDEN)
    range_hidden_idx = tl.arange(0, BLOCK_HIDDEN)

    logits_ptrs = Logits + range_batch_idx[:, None, None, None] * stride_logits_batch + range_seq_idx[None, :, None, None] * stride_logits_seq + range_head_idx[None, None, :, None] * stride_logits_head + range_hidden_idx[None, None, None, :] * stride_logits_hidden

    token_ids = tl.load(p_token_ids + range_token_idx, mask=mask_token, other=0)
    token_ids_count = tl.load(p_token_counts + range_token_idx, mask=mask_token, other=0)

    logits_freq = tl.load(logits_ptrs, mask=mask_token[None, None, None, :]).to(tl.float32)
    logits_freq *= coef_freq[None, None, None, :]

    logits_rep = tl.where(token_ids_count > 1, coef_rep[None, None, None, :] * (token_ids_count - 1), 0)

    logits_pre = tl.where(token_ids_count > 0, coef_pre[None, None, None, :] * token_ids_count, 0)

    logits_new = logits_freq + logits_rep + logits_pre

    tl.store(logits_ptrs, logits_new.to(Logits.dtype.element_ty), mask=mask_token[None, None, None, :])

def apply_penalty(Logits, cumsum_seq_len, penalties, token_ids, token_counts):
    assert Logits.is_contiguous()
    BLOCK = triton.next_power_of_2(max(Logits.shape[2:]))
    num_warps = 8

    Logits = Logits.contiguous()
    token_ids = token_ids.contiguous()
    token_counts = token_counts.contiguous()

    grid = (Logits.shape[0],)

    _fwd_kernel_apply_penalty[grid](
        *penalties,
        token_ids,
        token_counts,
        cumsum_seq_len,
        Logits,
        token_ids.stride(0),
        token_counts.stride(0),
        Logits.stride(0),
        Logits.stride(1),
        Logits.stride(2),
        Logits.stride(3),
        Logits.stride(2) + Logits.stride(3),
        Logits.stride(3),
        BLOCK_TOKEN=triton.next_power_of_2(token_ids.shape[0]),
        BLOCK_BATCH=1,
        BLOCK_HIDDEN=Logits.shape[2],
        BLOCK_SEQ=Logits.shape[1],
        num_warps=num_warps,
        num_stages=1,
    )
