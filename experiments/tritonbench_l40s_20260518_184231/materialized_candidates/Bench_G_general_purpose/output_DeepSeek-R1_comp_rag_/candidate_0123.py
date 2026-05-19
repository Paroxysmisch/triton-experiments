import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_apply_penalty(
    Logits, presence_penalty, frequency_penalty, repetition_penalty,
    p_token_ids, p_token_counts, p_cumsum_seq_len,
    stride_logit_b, stride_logit_s,
    BLOCK_P: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_freqency = tl.load(frequency_penalty + cur_batch)
    cur_presence = tl.load(presence_penalty + cur_batch)
    cur_repetition = tl.load(repetition_penalty + cur_batch)

    cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch)
    cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1)

    cur_batch_id_offset = cur_batch_start_index + tl.arange(0, BLOCK_P)
    mask = cur_batch_id_offset < cur_batch_end_index

    batch_ids = tl.load(p_token_ids + cur_batch_id_offset, mask=mask, other=0)
    batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset, mask=mask, other=0)
    
    row_start_ptr = Logits + cur_batch * stride_logit_b
    offsets = row_start_ptr + batch_ids
    cur_logits = tl.load(offsets, mask=mask, other=0.0)
    
    # Apply penalties in order: repetition (multiply), frequency (subtract count * freq), presence (subtract)
    adjusted_logits = cur_logits * cur_repetition
    adjusted_logits = adjusted_logits - batch_ids_count * cur_freqency
    adjusted_logits = adjusted_logits - cur_presence
    
    tl.store(offsets, adjusted_logits, mask=mask)

@torch.no_grad()
def apply_penalty(
    Logits, presence_penalty, frequency_penalty, repetition_penalty,
    p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch
):
    assert Logits.is_contiguous(), "Logits tensor must be contiguous"
    BLOCK = triton.next_power_of_2(p_max_len_in_batch)
    if BLOCK < 512:
        BLOCK = 512
    elif BLOCK > 1024:
        BLOCK = 1024
    num_warps = 8
    _fwd_kernel_apply_penalty[(Logits.shape[0],)](
        Logits, presence_penalty, frequency_penalty, repetition_penalty,
        p_token_ids, p_token_counts, p_cumsum_seq_len,
        Logits.stride(0), Logits.stride(1),
        num_warps=num_warps,
        BLOCK_P=BLOCK
    )
    return Logits
