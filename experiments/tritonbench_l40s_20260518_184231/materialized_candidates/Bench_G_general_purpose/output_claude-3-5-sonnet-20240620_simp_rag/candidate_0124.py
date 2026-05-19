import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_apply_penalty(
    # Pointers to input/output tensors
    Logits, presence_penalty, frequency_penalty,  # Note: fixed typo in frequency
    p_token_ids, p_token_counts, p_cumsum_seq_len,
    # Strides for accessing multi-dimensional tensors
    stride_logit_b, stride_logit_s,
    # Compile-time constants
    BLOCK_P: tl.constexpr
):
    # Get current batch index from program ID
    cur_batch = tl.program_id(0)
    
    # Load penalties for current batch
    cur_frequency = tl.load(frequency_penalty + cur_batch)
    cur_presence = tl.load(presence_penalty + cur_batch)
    
    # Get sequence boundaries for current batch
    start_idx = tl.load(p_cumsum_seq_len + cur_batch)
    end_idx = tl.load(p_cumsum_seq_len + cur_batch + 1)
    
    # Create offset array for current batch
    offsets = start_idx + tl.arange(0, BLOCK_P)
    mask = offsets < end_idx
    
    # Load token IDs and counts
    token_ids = tl.load(p_token_ids + offsets, mask=mask, other=0)
    token_counts = tl.load(p_token_counts + offsets, mask=mask, other=0)
    
    # Calculate base pointer for current batch's logits
    logits_base = Logits + cur_batch * stride_logit_b
    
    # Load logits for the tokens
    logits_ptr = logits_base + token_ids
    cur_logits = tl.load(logits_ptr, mask=mask, other=0.0)
    
    # Apply penalties:
    # 1. Frequency penalty: multiply count by penalty and subtract
    # 2. Presence penalty: subtract if token exists (count > 0)
    penalized_logits = cur_logits - (token_counts * cur_frequency) - cur_presence
    
    # Store results back to memory
    tl.store(logits_ptr, penalized_logits, mask=mask)

@torch.no_grad()
def apply_penalty(
    Logits, presence_penalty, frequency_penalty,
    p_token_ids, p_token_counts, p_cumsum_seq_len,
    p_max_len_in_batch
):
    """
    Wrapper function to apply penalties to logits
    
    Args:
        Logits: Input logits tensor (batch_size x vocab_size)
        presence_penalty: Penalty for token presence
        frequency_penalty: Penalty based on token frequency
        p_token_ids: Token IDs in the sequence
        p_token_counts: Count of each token's occurrence
        p_cumsum_seq_len: Cumulative sequence lengths
        p_max_len_in_batch: Maximum sequence length in batch
    """
    assert Logits.is_contiguous(), "Logits tensor must be contiguous"
    
    # Calculate optimal block size
    BLOCK = triton.next_power_of_2(p_max_len_in_batch)
    BLOCK = max(min(BLOCK, 1024), 512)  # Clamp between 512 and 1024
    
    # Launch kernel
    grid = (Logits.shape[0],)  # One thread block per batch
    _fwd_kernel_apply_penalty[grid](
        Logits, presence_penalty, frequency_penalty,
        p_token_ids, p_token_counts, p_cumsum_seq_len,
        Logits.stride(0), Logits.stride(1),
        num_warps=8,
        BLOCK_P=BLOCK
    )
    
    return Logits
