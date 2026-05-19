import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_apply_penalty(
    Logprobs, Logits, PresencePenalty, FrequencyPenalty, RepetitionPenalty, 
    p_token_ids, p_token_counts, p_cumsum_seq_len, p_batch_ids, p_seq_len,
    stride_logprobs_batch, stride_logprobs_logit,
    stride_logits_batch, stride_logits_logit,
    stride_presence_penalty_batch, stride_presence_penalty_logit,
    stride_frequency_penalty_batch, stride_frequency_penalty_logit,
    stride_repetition_penalty_batch, stride_repetition_penalty_logit,
    stride_token_ids_batch, stride_token_ids_token,
    stride_token_counts_batch, stride_token_counts_count,
    stride_cumsum_seq_len_batch, stride_cumsum_seq_len_seq_len,
    stride_batch_ids_batch, stride_batch_ids_id,
    num_batches, num_tokens,
    BLOCK_SIZE: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_token = tl.program_id(1)

    offs_logprobs = cur_batch * stride_logprobs_batch + cur_token * stride_logprobs_logit
    offs_logits = cur_batch * stride_logits_batch + cur_token * stride_logits_logit

    offs_freq_pen = cur_batch * stride_frequency_penalty_batch + cur_token * stride_frequency_penalty_logit
    offs_repetition_pen = cur_batch * stride_repetition_penalty_batch + cur_token * stride_repetition_penalty_logit
    offs_presence_pen = cur_batch * stride_presence_penalty_batch + cur_token * stride_presence_penalty_logit

    logprobs = tl.load(Logprobs + offs_logprobs)
    logits = tl.load(Logits + offs_logits)
    
    logit_pre_repetition = logits

    # repetition penalty
    logits = logits * tl.load(RepetitionPenalty + offs_repetition_pen)

    # frequency penalty
    freq_pen = tl.load(FrequencyPenalty + offs_freq_pen)
    logits = logits * (1 + freq_pen)

    # presence penalty
    presence_pen = tl.load(PresencePenalty + offs_presence_pen)
    logits = logits * (1 + presence_pen)

    # store
    tl.store(Logprobs + offs_logprobs, logprobs)
    tl.store(Logits + offs_logits, logits)

def apply_penalty(logprobs, logits, presence_penalty, frequency_penalty, repetition_penalty, 
                  p_token_ids, p_token_counts, p_cumsum_seq_len, p_token_ids_device='cuda', 
                  p_token_counts_device='cuda', p_cumsum_seq_len_device='cuda', 
                  p_batch_ids=None, p_seq_len=None, penalty_min=None, penalty_max=None, 
                  clamp_logits=True):
    assert logprobs.is_contiguous()
    assert logits.is_contiguous()

    assert p_token_ids.is_contiguous()
    assert p_token_counts.is_contiguous()
    assert p_cumsum_seq_len.is_contiguous()

    if p_batch_ids is not None:
        assert p_batch_ids.is_contiguous()
    if p_seq_len is not None:
        assert p_seq_len.is_contiguous()

    if p_token_ids_device != 'cuda':
        p_token_ids = p_token_ids.to('cuda')
    if p_token_counts_device != 'cuda':
        p_token_counts = p_token_counts.to('cuda')
    if p_cumsum_seq_len_device != 'cuda':
        p_cumsum_seq_len = p_cumsum_seq_len.to('cuda')

    num_batches = p_token_ids.size(0)
    num_tokens = p_token_ids.size(1)

    BLOCK = triton.next_power_of_2(max(torch.max(p_seq_len).item() if p_seq_len is not None else 0, 128))
    if p_seq_len is None:
        num_warps = 8
    else:
        num_warps = min(max(BLOCK // torch.max(p_seq_len).item(), 1), 8)

    _fwd_kernel_apply_penalty[(num_batches, num_tokens, )](
        logprobs, logits, presence_penalty, frequency_penalty, repetition_penalty, 
        p_token_ids, p_token_counts, p_cumsum_seq_len, p_batch_ids, p_seq_len,
        logprobs.stride(0), logprobs.stride(1),
        logits.stride(0), logits.stride(1),
        presence_penalty.stride(0), presence_penalty.stride(1),
        frequency_penalty.stride(0), frequency_penalty.stride(1),
        repetition_penalty.stride(0), repetition_penalty.stride(1),
        p_token_ids.stride(0), p_token_ids.stride(1),
        p_token_counts.stride(0), p_token_counts.stride(1),
        p_cumsum_seq_len.stride(0), p_cumsum_seq_len.stride(1),
        p_batch_ids.stride(0), p_batch_ids.stride(1) if p_batch_ids is not None else 0,
        num_batches, num_tokens,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK,
    )
    return
