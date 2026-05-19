import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_att2(
    Prob_ptr, V_ptr, Out_ptr, Req_to_tokens_ptr,
    B_req_idx_ptr, B_Start_Loc_ptr, B_Seqlen_ptr,
    stride_prob_batch, stride_prob_head, stride_prob_seq,
    stride_v_batch, stride_v_head, stride_v_seq, stride_v_dim,
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_dim,
    NUM_BATCHES, NUM_HEADS, SEQLEN, D_MODEL,
    BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    num_heads_per_batch = NUM_HEADS
    
    # Calculate current batch and head
    cur_batch = pid // num_heads_per_batch
    cur_head = pid % num_heads_per_batch
    
    # Get batch-specific information
    batch_req_idx = tl.load(B_req_idx_ptr + cur_batch)
    batch_start = tl.load(B_Start_Loc_ptr + batch_req_idx)
    batch_seq_len = tl.load(B_Seqlen_ptr + batch_req_idx)
    
    # Calculate offsets
    prob_offset = cur_batch * stride_prob_batch + cur_head * stride_prob_head
    v_offset = cur_batch * stride_v_batch + cur_head * stride_v_head
    out_offset = cur_batch * stride_out_batch + cur_head * stride_out_head
    
    # Create ranges for the block
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    # Loop over sequence length in blocks
    for seq_idx in range(0, batch_seq_len, BLOCK_N):
        # Compute number of valid elements in current block
        valid_n = tl.minimum(BLOCK_N, batch_seq_len - seq_idx)
        
        # Load probabilities
        token_idx = tl.load(Req_to_tokens_ptr + batch_start + seq_idx + offs_n,
                           mask=offs_n < valid_n, other=0.0)
        probs = tl.load(Prob_ptr + prob_offset + token_idx * stride_prob_seq,
                       mask=offs_n < valid_n, other=0.0)
        
        # Load values
        v_idx = v_offset + token_idx * stride_v_seq + offs_d * stride_v_dim
        values = tl.load(V_ptr + v_idx,
                        mask=offs_d < D_MODEL, other=0.0)
        
        # Compute weighted sum
        acc += tl.sum(probs[:, None] * values, axis=0)
    
    # Store result
    out_idx = out_offset + offs_d * stride_out_dim
    tl.store(Out_ptr + out_idx, acc, mask=offs_d < D_MODEL)

def token_att_fwd2(prob, v, req_to_tokens, b_req_idx, b_start_loc, b_seqlen):
    """
    Wrapper function for token attention forward pass
    
    Args:
        prob: attention probabilities [batch, num_heads, seq_len]
        v: value tensor [batch, num_heads, seq_len, d_model]
        req_to_tokens: mapping from requests to token indices
        b_req_idx: batch request indices
        b_start_loc: batch start locations
        b_seqlen: batch sequence lengths
    """
    batch, num_heads, seq_len = prob.shape
    d_model = v.shape[-1]
    
    # Output tensor
    out = torch.empty((batch, num_heads, d_model), 
                     device=prob.device, dtype=prob.dtype)
    
    # Configure block sizes
    BLOCK_DMODEL = 128
    BLOCK_N = 32
    
    # Launch kernel
    grid = (batch * num_heads,)
    _fwd_kernel_token_att2[grid](
        prob, v, out, req_to_tokens,
        b_req_idx, b_start_loc, b_seqlen,
        prob.stride(0), prob.stride(1), prob.stride(2),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), 1, out.stride(2),
        batch, num_heads, seq_len, d_model,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2
    )
    
    return out
