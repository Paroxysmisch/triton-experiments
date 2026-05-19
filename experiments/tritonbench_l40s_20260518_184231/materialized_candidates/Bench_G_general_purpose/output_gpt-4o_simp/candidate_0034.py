import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    prob_ptr,  # Pointer to the probability tensor
    value_ptr, # Pointer to the value tensor
    output_ptr, # Pointer to the output tensor
    batch_size, # Number of batches
    num_heads,  # Number of attention heads
    seq_len,    # Sequence length
    head_dim,   # Dimension of each head
    BLOCK_SIZE: tl.constexpr, # Block size for processing
):
    # Compute program ID
    pid = tl.program_id(0)
    
    # Calculate indices for the current block
    batch_id = pid // (num_heads * seq_len)
    head_id = (pid // seq_len) % num_heads
    token_id = pid % seq_len
    
    # Offset pointers to the start of the current batch and head
    prob_offset = batch_id * num_heads * seq_len * seq_len + head_id * seq_len * seq_len + token_id * seq_len
    value_offset = batch_id * num_heads * seq_len * head_dim + head_id * seq_len * head_dim
    output_offset = batch_id * num_heads * seq_len * head_dim + head_id * seq_len * head_dim + token_id * head_dim
    
    # Load probabilities and values
    probs = tl.load(prob_ptr + prob_offset + tl.arange(0, seq_len))
    values = tl.load(value_ptr + value_offset + tl.arange(0, head_dim))
    
    # Initialize accumulator
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    # Compute attention output
    for i in range(seq_len):
        value = tl.load(value_ptr + value_offset + i * head_dim + tl.arange(0, head_dim))
        acc += probs[i] * value
    
    # Store result
    tl.store(output_ptr + output_offset + tl.arange(0, head_dim), acc)

def token_att_fwd2(prob, value):
    batch_size, num_heads, seq_len, head_dim = value.shape
    
    # Allocate output tensor
    output = torch.empty((batch_size, num_heads, seq_len, head_dim), device=prob.device, dtype=prob.dtype)
    
    # Define grid dimensions
    grid = (batch_size * num_heads * seq_len,)
    
    # Launch Triton kernel
    _fwd_kernel_token_att2[grid](
        prob,
        value,
        output,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        BLOCK_SIZE=128  # Example block size
    )
    
    return output
