import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(
    Logits, V, Out,
    B_Loc, B_Start_Loc, B_Seqlen,
    stride_b, stride_h, stride_s, stride_d,
    BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Program ID for batch and head dimensions
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    # Compute base pointers
    logits_ptr = Logits + batch_id * stride_b + head_id * stride_h
    v_ptr = V + batch_id * stride_b + head_id * stride_h
    out_ptr = Out + batch_id * stride_b + head_id * stride_h
    
    # Load sequence information for this batch
    start_loc = tl.load(B_Start_Loc + batch_id)
    seqlen = tl.load(B_Seqlen + batch_id)
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    e_sum = 0.0
    
    # Process sequence in blocks
    for block_start in range(0, seqlen, BLOCK_N):
        block_size = tl.minimum(BLOCK_N, seqlen - block_start)
        
        # Create offsets for this block
        offs_n = block_start + tl.arange(0, block_size)
        
        # Load indices and mask
        loc_idx = tl.load(B_Loc + start_loc + offs_n)
        mask = offs_n < seqlen
        
        # Load logits and values
        logits = tl.load(logits_ptr + loc_idx * stride_s, mask=mask, other=-float('inf'))
        
        # Compute max for numerical stability
        e_max = tl.maximum(tl.max(logits, axis=0), -float('inf'))
        
        # Compute exponentials
        p = tl.exp(logits - e_max)
        e_sum += tl.sum(p, axis=0)
        
        # Load values and compute weighted sum
        offs_d = tl.arange(0, BLOCK_DMODEL)
        v = tl.load(v_ptr + loc_idx[:, None] * stride_s + offs_d[None, :] * stride_d,
                   mask=mask[:, None], other=0.0)
        
        # Accumulate weighted values
        acc += tl.sum(p[:, None] * v, axis=0)
    
    # Normalize and store result
    acc = acc / e_sum
    
    # Store final output
    offs_d = tl.arange(0, BLOCK_DMODEL)
    tl.store(out_ptr + offs_d * stride_d, acc)

# Wrapper function
def token_softmax_reducev_fwd(logits, values, batch_loc, batch_start_loc, batch_seqlen):
    """
    Forward pass for token-wise softmax reduction.
    
    Args:
        logits: attention scores [B, H, S]
        values: value tensors [B, H, S, D]
        batch_loc: token indices
        batch_start_loc: starting indices for each batch
        batch_seqlen: sequence lengths for each batch
    """
    batch_size, num_heads, max_seqlen = logits.shape
    _, _, _, d_model = values.shape
    
    # Output tensor
    output = torch.empty_like(values)
    
    # Configure block sizes
    BLOCK_N = 128
    BLOCK_DMODEL = d_model
    
    # Calculate strides
    stride_b = logits.stride(0)
    stride_h = logits.stride(1)
    stride_s = logits.stride(2)
    stride_d = values.stride(3)
    
    # Launch kernel
    grid = (batch_size, num_heads)
    _fwd_kernel[grid](
        logits, values, output,
        batch_loc, batch_start_loc, batch_seqlen,
        stride_b, stride_h, stride_s, stride_d,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2
    )
    
    return output
