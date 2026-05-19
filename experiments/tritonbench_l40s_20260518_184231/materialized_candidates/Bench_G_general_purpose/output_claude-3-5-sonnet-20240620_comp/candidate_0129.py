import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_apply_penalty(
    # Pointers to tensors
    p_logits, p_token_ids, p_token_counts, p_cumsum_seq_len,
    # Penalty coefficients 
    presence_penalty, frequency_penalty, repetition_penalty,
    # Tensor metadata
    cur_batch, seq_len, vocab_size,
    stride_b, stride_s, stride_v,
    BLOCK: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Load sequence range for current batch
    start_idx = tl.load(p_cumsum_seq_len + cur_batch - 1) if cur_batch > 0 else 0
    end_idx = tl.load(p_cumsum_seq_len + cur_batch)
    seq_size = end_idx - start_idx
    
    # Each thread processes BLOCK tokens
    for idx in range(pid * BLOCK, tl.minimum((pid + 1) * BLOCK, vocab_size)):
        # Load logit value
        logit_ptr = p_logits + cur_batch * stride_b + idx * stride_v
        logit = tl.load(logit_ptr)
        
        # Initialize penalty multiplier
        penalty = 1.0
        
        # Check if token exists in sequence history
        for seq_idx in range(start_idx, end_idx):
            token_id = tl.load(p_token_ids + seq_idx)
            token_count = tl.load(p_token_counts + seq_idx)
            
            # Apply penalties if token matches
            is_match = token_id == idx
            if is_match:
                # Repetition penalty
                penalty *= tl.where(token_count > 0, repetition_penalty, 1.0)
                # Frequency penalty
                logit -= frequency_penalty * token_count
                # Presence penalty
                logit -= presence_penalty
        
        # Apply final penalty and store result
        logit = tl.where(logit >= 0, logit * penalty, logit / penalty)
        tl.store(logit_ptr, logit)

def apply_penalty(
    Logits: torch.Tensor,
    token_ids: torch.Tensor,
    token_counts: torch.Tensor,
    cumsum_seq_len: torch.Tensor,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Apply presence, frequency and repetition penalties to logits.
    
    Args:
        Logits: Tensor of shape [batch_size, vocab_size] containing logits
        token_ids: Tensor containing token IDs from sequence history
        token_counts: Tensor containing token counts
        cumsum_seq_len: Cumulative sum of sequence lengths per batch
        presence_penalty: Penalty for token presence in history
        frequency_penalty: Penalty based on token frequency
        repetition_penalty: Penalty for repeated tokens
        
    Returns:
        Tensor of adjusted logits
    """
    # Ensure contiguous tensors
    Logits = Logits.contiguous()
    
    # Get dimensions
    batch_size, vocab_size = Logits.shape
    
    # Calculate optimal block size (power of 2, minimum 32)
    BLOCK = max(triton.next_power_of_2(vocab_size // 256), 32)
    
    # Calculate grid size
    grid = (triton.cdiv(vocab_size, BLOCK),)
    
    # Get strides
    stride_b, stride_v = Logits.stride()
    
    # Launch kernel for each batch
    for batch_idx in range(batch_size):
        _fwd_kernel_apply_penalty[grid](
            Logits, token_ids, token_counts, cumsum_seq_len,
            presence_penalty, frequency_penalty, repetition_penalty,
            batch_idx, cumsum_seq_len.shape[0], vocab_size,
            stride_b, 1, stride_v,
            BLOCK=BLOCK,
            num_warps=8,
        )
    
    return Logits
