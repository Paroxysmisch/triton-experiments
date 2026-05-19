import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_softmax(
    # Pointers to tensors
    Logits_ptr, B_Start_Loc_ptr, B_Seqlen_ptr, Prob_Out_ptr,
    # Dimensions
    batch_size, num_heads, max_seqlen,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate batch and head indices
    batch_id = pid // num_heads
    head_id = pid % num_heads
    
    # Load sequence start location and length for this batch
    b_start = tl.load(B_Start_Loc_ptr + batch_id)
    seqlen = tl.load(B_Seqlen_ptr + batch_id)
    
    # Initialize max value and accumulator
    max_val = float("-inf")
    acc = 0.0
    
    # Compute max for numerical stability
    for idx in range(0, seqlen, BLOCK_SIZE):
        block_size = min(BLOCK_SIZE, seqlen - idx)
        col = idx + tl.arange(0, block_size)
        
        # Load logits
        offset = b_start * num_heads * max_seqlen + head_id * max_seqlen + col
        logits = tl.load(Logits_ptr + offset, mask=col < seqlen, other=float("-inf"))
        
        # Update max
        max_val = tl.maximum(max_val, tl.max(logits, axis=0))
    
    # Compute exponentials and sum
    for idx in range(0, seqlen, BLOCK_SIZE):
        block_size = min(BLOCK_SIZE, seqlen - idx)
        col = idx + tl.arange(0, block_size)
        
        # Load logits
        offset = b_start * num_heads * max_seqlen + head_id * max_seqlen + col
        logits = tl.load(Logits_ptr + offset, mask=col < seqlen, other=float("-inf"))
        
        # Apply exp(logits - max_val)
        exp_val = tl.exp(logits - max_val)
        acc += tl.sum(exp_val, axis=0)
        
        # Store intermediate results
        tl.store(Prob_Out_ptr + offset, exp_val, mask=col < seqlen)
    
    # Normalize by sum
    for idx in range(0, seqlen, BLOCK_SIZE):
        block_size = min(BLOCK_SIZE, seqlen - idx)
        col = idx + tl.arange(0, block_size)
        
        offset = b_start * num_heads * max_seqlen + head_id * max_seqlen + col
        probs = tl.load(Prob_Out_ptr + offset, mask=col < seqlen, other=0.0)
        probs = probs / acc
        tl.store(Prob_Out_ptr + offset, probs, mask=col < seqlen)

def token_softmax_fwd(logits, batch_start_loc, batch_seqlens):
    """
    Forward pass for token-wise softmax with variable sequence lengths.
    
    Args:
        logits: Input tensor of shape [total_tokens, num_heads, max_seqlen]
        batch_start_loc: Tensor containing start indices for each batch
        batch_seqlens: Tensor containing sequence lengths for each batch
        
    Returns:
        Probability tensor of same shape as logits
    """
    batch_size = batch_start_loc.shape[0]
    num_heads = logits.shape[1]
    max_seqlen = logits.shape[-1]
    
    # Allocate output tensor
    prob_out = torch.empty_like(logits)
    
    # Launch kernel
    BLOCK_SIZE = 128
    grid = (batch_size * num_heads,)
    
    _fwd_kernel_token_softmax[grid](
        logits, batch_start_loc, batch_seqlens, prob_out,
        batch_size, num_heads, max_seqlen,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return prob_out
