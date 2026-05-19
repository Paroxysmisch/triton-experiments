import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_apply_penalty(
    # Pointers to tensors
    logits_ptr,           # [B, V] - pointer to logits tensor
    token_ids_ptr,        # [B, S] - pointer to token IDs
    token_counts_ptr,     # [B, V] - pointer to token counts
    seq_lens_ptr,         # [B] - pointer to sequence lengths
    presence_penalty_ptr, # [B] - pointer to presence penalties
    freq_penalty_ptr,     # [B] - pointer to frequency penalties
    rep_penalty_ptr,      # [B] - pointer to repetition penalties
    # Tensor dimensions
    batch_size,
    vocab_size,
    max_seq_len,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    batch_idx = pid // vocab_size
    vocab_idx = pid % vocab_size

    # Exit if we're out of bounds
    if batch_idx >= batch_size:
        return

    # Load sequence length for this batch
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    
    # Load penalties for this batch
    presence_penalty = tl.load(presence_penalty_ptr + batch_idx)
    freq_penalty = tl.load(freq_penalty_ptr + batch_idx)
    rep_penalty = tl.load(rep_penalty_ptr + batch_idx)

    # Load token count for current token
    token_count = tl.load(token_counts_ptr + batch_idx * vocab_size + vocab_idx)
    
    # Load current logit value
    logit = tl.load(logits_ptr + batch_idx * vocab_size + vocab_idx)
    
    # Apply presence penalty
    if token_count > 0:
        logit = logit - presence_penalty
    
    # Apply frequency penalty
    logit = logit - freq_penalty * token_count
    
    # Apply repetition penalty
    if token_count > 0:
        if logit > 0:
            logit = logit / rep_penalty
        else:
            logit = logit * rep_penalty
    
    # Store result
    tl.store(logits_ptr + batch_idx * vocab_size + vocab_idx, logit)

def apply_penalty(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    token_counts: torch.Tensor,
    sequence_lengths: torch.Tensor,
    presence_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    repetition_penalty: torch.Tensor,
) -> torch.Tensor:
    """
    Apply presence, frequency, and repetition penalties to logits.
    
    Args:
        logits: shape [batch_size, vocab_size] - Input logits
        token_ids: shape [batch_size, seq_len] - Token IDs in the sequence
        token_counts: shape [batch_size, vocab_size] - Count of each token
        sequence_lengths: shape [batch_size] - Length of each sequence
        presence_penalty: shape [batch_size] - Presence penalty per batch
        frequency_penalty: shape [batch_size] - Frequency penalty per batch
        repetition_penalty: shape [batch_size] - Repetition penalty per batch
    
    Returns:
        torch.Tensor: Penalized logits with same shape as input
    """
    batch_size, vocab_size = logits.shape
    max_seq_len = token_ids.shape[1]
    
    # Ensure all inputs are on the same device
    device = logits.device
    token_ids = token_ids.to(device)
    token_counts = token_counts.to(device)
    sequence_lengths = sequence_lengths.to(device)
    presence_penalty = presence_penalty.to(device)
    frequency_penalty = frequency_penalty.to(device)
    repetition_penalty = repetition_penalty.to(device)
    
    # Launch kernel
    grid = (batch_size * vocab_size,)
    _fwd_kernel_apply_penalty[grid](
        logits,
        token_ids,
        token_counts,
        sequence_lengths,
        presence_penalty,
        frequency_penalty,
        repetition_penalty,
        batch_size,
        vocab_size,
        max_seq_len,
        BLOCK_SIZE=128,
    )
    
    return logits
