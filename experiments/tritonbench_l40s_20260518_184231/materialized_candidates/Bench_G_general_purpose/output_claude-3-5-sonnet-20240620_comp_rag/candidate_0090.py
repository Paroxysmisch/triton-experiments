import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Out_ptr, V_ptr, Logits_ptr,
    B_Loc_ptr, B_Start_Loc_ptr, B_Seqlen_ptr,
    stride_ob, stride_oh, stride_os,  # Out strides
    stride_vb, stride_vh, stride_vs,  # V strides
    stride_lb, stride_lh, stride_ls,  # Logits strides
    B, H, max_seqlen,
    BLOCK_N: tl.constexpr
):
    # Program ID
    bid = tl.program_id(0)  # Batch index
    hid = tl.program_id(1)  # Head index
    
    # Compute pointer offsets
    b_loc = tl.load(B_Loc_ptr + bid)
    b_start = tl.load(B_Start_Loc_ptr + bid)
    b_seqlen = tl.load(B_Seqlen_ptr + bid)
    
    # Initialize pointers
    offs_n = tl.arange(0, BLOCK_N)
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    e_max = tl.zeros([1], dtype=tl.float32) - float('inf')
    e_sum = tl.zeros([1], dtype=tl.float32)
    
    # Load base pointers
    logits_base = Logits_ptr + b_loc * stride_lb + hid * stride_lh
    v_base = V_ptr + b_loc * stride_vb + hid * stride_vh
    
    # Main loop over sequence length
    for n in range(0, b_seqlen, BLOCK_N):
        mask = offs_n < (b_seqlen - n)
        
        # Load logits and compute max for numerical stability
        logits_ptrs = logits_base + (n + offs_n) * stride_ls
        logits = tl.load(logits_ptrs, mask=mask, other=-float('inf'))
        block_max = tl.max(logits, axis=0)
        e_max = tl.maximum(e_max, block_max)
        
        # Compute exponentials
        logits = logits - e_max
        p = tl.exp(logits)
        e_sum += tl.sum(p, axis=0)
        
        # Load values and accumulate
        v_ptrs = v_base + (n + offs_n) * stride_vs
        v = tl.load(v_ptrs, mask=mask, other=0.0)
        acc += p[:, None] * v
    
    # Normalize
    acc = acc / e_sum
    
    # Store results
    out_ptr = Out_ptr + b_loc * stride_ob + hid * stride_oh
    for n in range(0, BLOCK_N):
        if n < b_seqlen:
            tl.store(out_ptr + n * stride_os, acc[n])

def token_softmax_reducev_fwd(logits, v, b_loc, b_start_loc, b_seqlen):
    """
    Wrapper function for the token-wise softmax reduction kernel
    """
    batch, heads, max_seqlen = logits.shape
    _, _, dim = v.shape
    
    # Allocate output
    out = torch.empty_like(v)
    
    # Configure kernel parameters
    BLOCK_N = triton.next_power_of_2(max_seqlen)
    if BLOCK_N > 2048:
        BLOCK_N = 2048
    
    # Launch kernel
    grid = (batch, heads)
    _fwd_kernel[grid](
        out, v, logits,
        b_loc, b_start_loc, b_seqlen,
        out.stride(0), out.stride(1), out.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        logits.stride(0), logits.stride(1), logits.stride(2),
        batch, heads, max_seqlen,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2
    )
    
    return out
