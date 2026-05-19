import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_apply_penalty(
    Logits,
    p_presence_penalty,
    p_frequency_penalty,
    p_repetition_penalty,
    p_token_ids,
    p_token_counts,
    p_cumsum_seq_len,
    stride_batch,
    vocab_size,
    BLOCK: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    
    # Load penalty coefficients for current batch
    presence_pen = tl.load(p_presence_penalty + cur_batch)
    freq_pen = tl.load(p_frequency_penalty + cur_batch)
    rep_pen = tl.load(p_repetition_penalty + cur_batch)
    
    # Get token range for current batch
    offset_start = tl.load(p_cumsum_seq_len + cur_batch)
    offset_end = tl.load(p_cumsum_seq_len + cur_batch + 1) - 1
    num_tokens = offset_end - offset_start + 1
    
    logits_ptr = Logits + cur_batch * stride_batch
    
    # Process vocabulary in blocks
    for idx in range(0, vocab_size, BLOCK):
        vocab_indices = idx + tl.arange(0, BLOCK)
        mask = vocab_indices < vocab_size
        
        # Load current block of logits
        current_logits = tl.load(logits_ptr + vocab_indices, mask=mask, other=0)
        
        # Apply penalties for each historical token
        for i in range(num_tokens):
            token_id = tl.load(p_token_ids + offset_start + i)
            count = tl.load(p_token_counts + offset_start + i)
            
            # Create mask for matching tokens
            match_mask = (vocab_indices == token_id) & mask
            
            # Apply penalties in sequence
            if rep_pen != 1.0:
                rep_factor = rep_pen ** count
                current_logits = tl.where(match_mask, current_logits * rep_factor, current_logits)
            
            current_logits = tl.where(match_mask, current_logits - freq_pen * count, current_logits)
            current_logits = tl.where(match_mask, current_logits - presence_pen, current_logits)
        
        # Store modified logits
        tl.store(logits_ptr + vocab_indices, current_logits, mask=mask)

def apply_penalty(
    Logits: torch.Tensor,
    presence_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    repetition_penalty: torch.Tensor,
    token_ids: torch.Tensor,
    token_counts: torch.Tensor,
    cumsum_seq_len: torch.Tensor,
):
    # Ensure contiguous memory access
    Logits = Logits.contiguous()
    vocab_size = Logits.size(-1)
    batch_size = Logits.size(0)
    
    # Determine optimal block size
    BLOCK = triton.next_power_of_2(vocab_size)
    BLOCK = max(BLOCK, 1024)
    
    # Launch kernel with fixed 8 warps
    grid = (batch_size,)
    _fwd_kernel_apply_penalty[grid](
        Logits,
        presence_penalty,
        frequency_penalty,
        repetition_penalty,
        token_ids,
        token_counts,
        cumsum_seq_len,
        Logits.stride(0),
        vocab_size,
        BLOCK=BLOCK,
        num_warps=8,
    )
