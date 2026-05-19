import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_att1(
    # Pointers to matrices
    q_ptr, k_ptr, att_out_ptr,
    # Location information
    B_Loc_ptr, B_Start_Loc_ptr, B_Seqlen_ptr,
    # Matrix dimensions
    batch_size, head_size, max_input_len,
    # Scaling factor for attention
    scale,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Batch and sequence index
    batch_idx = pid // max_input_len
    seq_idx = pid % max_input_len
    
    # Load sequence information
    b_start = tl.load(B_Start_Loc_ptr + batch_idx)
    b_seqlen = tl.load(B_Seqlen_ptr + batch_idx)
    
    # Check if current position is valid
    if seq_idx >= b_seqlen:
        return
        
    # Initialize accumulator for dot product
    acc = 0.0
    
    # Load query vector for current position
    q_offset = (b_start + seq_idx) * head_size
    
    # Block-level loop for loading query elements
    for block_start in range(0, head_size, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, head_size)
        
        # Load query block
        q = tl.load(q_ptr + q_offset + block_start + tl.arange(0, block_end - block_start))
        
        # Load key block and compute partial dot product
        k_offset = (b_start + seq_idx) * head_size
        k = tl.load(k_ptr + k_offset + block_start + tl.arange(0, block_end - block_start))
        
        # Accumulate dot product
        acc += tl.sum(q * k)
    
    # Scale the attention score
    acc = acc * scale
    
    # Store the result
    att_out_offset = b_start * max_input_len + seq_idx
    tl.store(att_out_ptr + att_out_offset, acc)

def token_att_fwd(q, k, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
    """
    Wrapper function for token attention forward pass
    
    Args:
        q: Query tensor [total_tokens, head_size]
        k: Key tensor [total_tokens, head_size]
        B_Loc: Batch location information
        B_Start_Loc: Starting locations for each sequence
        B_Seqlen: Sequence lengths
        max_input_len: Maximum input sequence length
    """
    batch_size = B_Seqlen.shape[0]
    head_size = q.shape[1]
    
    # Allocate output tensor
    att_out = torch.empty((B_Start_Loc[-1] * max_input_len,), 
                         device=q.device, dtype=q.dtype)
    
    # Calculate scaling factor
    scale = 1.0 / (head_size ** 0.5)
    
    # Configure grid and block sizes
    grid = (batch_size * max_input_len,)
    
    # Determine optimal block size (power of 2)
    BLOCK_SIZE = min(triton.next_power_of_2(head_size), 128)
    
    # Launch kernel
    _fwd_kernel_token_att1[grid](
        q, k, att_out,
        B_Loc, B_Start_Loc, B_Seqlen,
        batch_size, head_size, max_input_len,
        scale,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return att_out
