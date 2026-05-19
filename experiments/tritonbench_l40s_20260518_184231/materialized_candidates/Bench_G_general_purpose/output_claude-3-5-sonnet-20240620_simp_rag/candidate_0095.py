import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Out, V, Logics,
    B_Loc, B_Start_Loc, B_Seqlen,
    stride_ob, stride_oh, stride_os,
    stride_vb, stride_vh, stride_vs,
    stride_lb, stride_lh, stride_ls,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID gives us batch and head position
    pid = tl.program_id(0)
    batch_id = pid // stride_oh  # Current batch
    head_id = pid % stride_oh    # Current head
    
    # Get sequence length for current batch
    seq_len = tl.load(B_Seqlen + batch_id)
    batch_start = tl.load(B_Start_Loc + batch_id)
    
    # Initialize pointers for current batch and head
    offs_v = batch_id * stride_vb + head_id * stride_vh
    offs_l = batch_id * stride_lb + head_id * stride_lh
    offs_o = batch_id * stride_ob + head_id * stride_oh
    
    # Process sequence in blocks
    for start_n in range(0, seq_len, BLOCK_SIZE):
        # Load block of logits
        offs_n = tl.arange(0, BLOCK_SIZE)
        mask = offs_n < (seq_len - start_n)
        
        # Load logits and compute max for stability
        l_ptr = Logics + offs_l + (start_n + offs_n) * stride_ls
        logits = tl.load(l_ptr, mask=mask, other=-float('inf'))
        max_logits = tl.max(logits, axis=0)
        
        # Compute softmax numerator (exp(logits - max))
        numerator = tl.exp(logits - max_logits)
        denominator = tl.sum(numerator, axis=0)
        
        # Load values and compute weighted sum
        v_ptr = V + offs_v + (start_n + offs_n) * stride_vs
        values = tl.load(v_ptr, mask=mask, other=0.0)
        
        # Compute output
        softmax_output = numerator / denominator
        weighted_values = values * softmax_output
        
        # Store results
        o_ptr = Out + offs_o + (start_n + offs_n) * stride_os
        tl.store(o_ptr, weighted_values, mask=mask)

def token_softmax_reducev_fwd(logits, values, batch_info):
    """
    Wrapper function for token softmax reduction forward pass
    Args:
        logits: attention logits tensor [batch, heads, seqlen, seqlen]
        values: value tensor [batch, heads, seqlen, hidden]
        batch_info: tuple of (B_Loc, B_Start_Loc, B_Seqlen) tensors
    Returns:
        output tensor of same shape as values
    """
    batch_size, num_heads, seq_len, hidden_dim = values.shape
    output = torch.empty_like(values)
    
    # Extract batch info tensors
    B_Loc, B_Start_Loc, B_Seqlen = batch_info
    
    # Calculate strides
    stride_ob = output.stride(0)
    stride_oh = output.stride(1)
    stride_os = output.stride(2)
    
    stride_vb = values.stride(0)
    stride_vh = values.stride(1)
    stride_vs = values.stride(2)
    
    stride_lb = logits.stride(0)
    stride_lh = logits.stride(1)
    stride_ls = logits.stride(2)
    
    # Compute optimal block size
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    
    # Launch kernel
    grid = (batch_size * num_heads,)
    _fwd_kernel[grid](
        output, values, logits,
        B_Loc, B_Start_Loc, B_Seqlen,
        stride_ob, stride_oh, stride_os,
        stride_vb, stride_vh, stride_vs,
        stride_lb, stride_lh, stride_ls,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
