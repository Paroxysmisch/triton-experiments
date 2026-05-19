import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_softmax(
    Logits, B_Start_Loc, B_Seqlen, Prob_Out,
    stride_b, stride_h, stride_s,  # Strides for batch, head, and sequence dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and head indices
    pid = tl.program_id(0)
    batch_id = pid // stride_h
    head_id = pid % stride_h

    # Get sequence info for this batch
    seq_start = tl.load(B_Start_Loc + batch_id)
    seq_len = tl.load(B_Seqlen + batch_id)

    # Compute base pointers
    base_offset = batch_id * stride_b + head_id * stride_s
    logits_ptr = Logits + base_offset + seq_start
    output_ptr = Prob_Out + base_offset + seq_start

    # Initialize variables for max computation
    max_val = float("-inf")
    
    # STEP 1: Find max value for numerical stability
    for block_start in range(0, seq_len, BLOCK_SIZE):
        block_size = min(BLOCK_SIZE, seq_len - block_start)
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < seq_len
        
        # Load logits block
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=float("-inf"))
        max_val = tl.maximum(max_val, tl.max(logits, axis=0))

    # STEP 2: Compute exponentials and sum
    sum_exp = 0.0
    for block_start in range(0, seq_len, BLOCK_SIZE):
        block_size = min(BLOCK_SIZE, seq_len - block_start)
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < seq_len
        
        # Load and compute exponentials
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=float("-inf"))
        exp_vals = tl.exp(logits - max_val)
        sum_exp += tl.sum(exp_vals * mask, axis=0)

    # STEP 3: Compute softmax probabilities
    for block_start in range(0, seq_len, BLOCK_SIZE):
        block_size = min(BLOCK_SIZE, seq_len - block_start)
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < seq_len
        
        # Load, compute softmax, and store results
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=float("-inf"))
        probs = tl.exp(logits - max_val) / sum_exp
        tl.store(output_ptr + col_offsets, probs, mask=mask)

@torch.no_grad()
def token_softmax_fwd(logits, batch_start_loc, batch_seqlens):
    """
    Wrapper function for token softmax forward pass
    
    Args:
        logits: Input tensor of shape [total_tokens, num_heads, hidden_dim]
        batch_start_loc: Tensor containing starting indices for each batch
        batch_seqlens: Tensor containing sequence lengths for each batch
    
    Returns:
        Tensor containing softmax probabilities
    """
    batch_size = batch_seqlens.shape[0]
    num_heads = logits.shape[1]
    max_seq_len = torch.max(batch_seqlens).item()
    
    # Compute optimal block size (power of 2)
    BLOCK_SIZE = triton.next_power_of_2(max_seq_len)
    BLOCK_SIZE = min(BLOCK_SIZE, 512)  # Cap at 512 for efficiency
    
    # Compute number of warps based on block size
    num_warps = 4
    if BLOCK_SIZE >= 256:
        num_warps = 8
    elif BLOCK_SIZE >= 128:
        num_warps = 4
    
    # Initialize output tensor
    output = torch.empty_like(logits)
    
    # Launch kernel
    grid = (batch_size * num_heads,)
    _fwd_kernel_token_softmax[grid](
        logits, batch_start_loc, batch_seqlens, output,
        logits.stride(0), num_heads, 1,  # Strides
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return output
