import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_apply_penalty(
    Logits,  # Pointer to the logits tensor
    p_token_ids,  # Pointer to the token IDs tensor
    p_token_counts,  # Pointer to the token counts tensor
    p_cumsum_seq_len,  # Pointer to the cumulative sequence lengths tensor
    p_presence_penalty,  # Pointer to the presence penalty tensor
    p_frequency_penalty,  # Pointer to the frequency penalty tensor
    p_repetition_penalty,  # Pointer to the repetition penalty tensor
    batch_size,  # Number of batches
    seq_len,  # Sequence length
    vocab_size,  # Vocabulary size
    BLOCK: tl.constexpr,  # Block size
):
    cur_batch = tl.program_id(0)
    if cur_batch >= batch_size:
        return

    # Load penalty coefficients for the current batch
    presence_penalty = tl.load(p_presence_penalty + cur_batch)
    frequency_penalty = tl.load(p_frequency_penalty + cur_batch)
    repetition_penalty = tl.load(p_repetition_penalty + cur_batch)

    # Determine the range of token indices for the current batch
    start_idx = tl.load(p_cumsum_seq_len + cur_batch)
    end_idx = tl.load(p_cumsum_seq_len + cur_batch + 1)

    # Load token IDs and their corresponding counts
    batch_ids = tl.load(p_token_ids + start_idx:end_idx, mask=start_idx + tl.arange(0, BLOCK) < end_idx, other=0)
    batch_ids_count = tl.load(p_token_counts + start_idx:end_idx, mask=start_idx + tl.arange(0, BLOCK) < end_idx, other=0)

    # Adjust logits based on repetition, frequency, and presence penalties
    for i in range(end_idx - start_idx):
        token_id = batch_ids[i]
        token_count = batch_ids_count[i]

        # Repetition penalty
        if token_count > 0:
            if token_count == 1:
                logit = tl.load(Logits + cur_batch * vocab_size + token_id)
                adjusted_logit = logit / repetition_penalty
                tl.store(Logits + cur_batch * vocab_size + token_id, adjusted_logit)
            else:
                logit = tl.load(Logits + cur_batch * vocab_size + token_id)
                adjusted_logit = logit * repetition_penalty
                tl.store(Logits + cur_batch * vocab_size + token_id, adjusted_logit)

        # Frequency penalty
        logit = tl.load(Logits + cur_batch * vocab_size + token_id)
        adjusted_logit = logit - token_count * frequency_penalty
        tl.store(Logits + cur_batch * vocab_size + token_id, adjusted_logit)

        # Presence penalty
        if token_count > 0:
            logit = tl.load(Logits + cur_batch * vocab_size + token_id)
            adjusted_logit = logit - presence_penalty
            tl.store(Logits + cur_batch * vocab_size + token_id, adjusted_logit)

import torch
import triton
import triton.language as tl

def apply_penalty(
    Logits,  # Tensor of shape (batch_size, vocab_size)
    token_ids,  # Tensor of shape (total_tokens)
    token_counts,  # Tensor of shape (total_tokens)
    cumsum_seq_len,  # Tensor of shape (batch_size + 1)
    presence_penalty,  # Tensor of shape (batch_size)
    frequency_penalty,  # Tensor of shape (batch_size)
    repetition_penalty,  # Tensor of shape (batch_size)
):
    # Ensure Logits is contiguous
    Logits = Logits.contiguous()

    # Calculate the appropriate block size
    BLOCK = triton.next_power_of_2(Logits.shape[1])

    # Ensure BLOCK is at least 128 for efficiency
    BLOCK = max(BLOCK, 128)

    # Launch the kernel
    grid = (Logits.shape[0],)
    _fwd_kernel_apply_penalty[grid](
        Logits,  # Pointer to the logits tensor
        token_ids,  # Pointer to the token IDs tensor
        token_counts,  # Pointer to the token counts tensor
        cumsum_seq_len,  # Pointer to the cumulative sequence lengths tensor
        presence_penalty,  # Pointer to the presence penalty tensor
        frequency_penalty,  # Pointer to the frequency penalty tensor
        repetition_penalty,  # Pointer to the repetition penalty tensor
        Logits.shape[0],  # Number of batches
        token_ids.shape[0],  # Sequence length
        Logits.shape[1],  # Vocabulary size
        BLOCK,  # Block size
        num_warps=8  # Fixed number of warps
    )
