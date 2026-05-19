import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_softmax(
    Logits, Prob_Out,
    stride_b, stride_h, stride_s,
    n_batches, n_heads, seq_len,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute batch and head indices
    batch_id = pid // n_heads
    head_id = pid % n_heads
    
    # Compute base pointer for this (batch, head)
    base_ptr = batch_id * stride_b + head_id * stride_h
    
    # Load logits and compute max for numerical stability
    max_val = float("-inf")
    for start_s in range(0, seq_len, BLOCK_SIZE):
        # Handle boundary conditions
        size_s = min(BLOCK_SIZE, seq_len - start_s)
        
        # Load block of logits
        offset = base_ptr + start_s * stride_s
        logits = tl.load(Logits + offset, mask=start_s < seq_len, other=float("-inf"))
        
        # Update max
        max_val = tl.maximum(max_val, tl.max(logits, axis=0))
    
    # Compute exponentials and sum
    sum_exp = 0.0
    for start_s in range(0, seq_len, BLOCK_SIZE):
        size_s = min(BLOCK_SIZE, seq_len - start_s)
        
        # Load and subtract max for numerical stability
        offset = base_ptr + start_s * stride_s
        logits = tl.load(Logits + offset, mask=start_s < seq_len, other=float("-inf"))
        logits = logits - max_val
        
        # Compute exponentials
        exp_vals = tl.exp(logits)
        sum_exp += tl.sum(exp_vals, axis=0)
    
    # Compute softmax and write results
    for start_s in range(0, seq_len, BLOCK_SIZE):
        size_s = min(BLOCK_SIZE, seq_len - start_s)
        
        # Load logits again
        offset = base_ptr + start_s * stride_s
        logits = tl.load(Logits + offset, mask=start_s < seq_len, other=float("-inf"))
        logits = logits - max_val
        
        # Compute final probabilities
        probs = tl.exp(logits) / sum_exp
        
        # Store results
        tl.store(Prob_Out + offset, probs, mask=start_s < seq_len)

def token_softmax_fwd(logits):
    """
    Compute softmax probabilities for token logits.
    
    Args:
        logits: Input tensor of shape (batch_size, n_heads, seq_len)
        
    Returns:
        Tensor of same shape containing softmax probabilities
    """
    batch_size, n_heads, seq_len = logits.shape
    
    # Allocate output tensor
    prob_out = torch.empty_like(logits)
    
    # Calculate memory strides
    stride_b = logits.stride(0)
    stride_h = logits.stride(1)
    stride_s = logits.stride(2)
    
    # Configure block size based on sequence length
    BLOCK_SIZE = min(128, triton.next_power_of_2(seq_len))
    
    # Launch kernel
    grid = (batch_size * n_heads,)
    _fwd_kernel_token_softmax[grid](
        logits, prob_out,
        stride_b, stride_h, stride_s,
        batch_size, n_heads, seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return prob_out
