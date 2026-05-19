import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    # Pointers to matrices
    prob_ptr, v_ptr, out_ptr,
    # Matrix dimensions
    batch, heads, seqlen, dim,
    # Other parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    window_size: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Batch and head index
    batch_id = pid // heads
    head_id = pid % heads
    
    # Initialize offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load the starting position for this window
    start_m = (pid * BLOCK_M) % seqlen
    
    # Compute window boundaries
    window_start = tl.maximum(0, start_m - window_size)
    window_end = tl.minimum(seqlen, start_m + BLOCK_M + window_size)
    
    # Pointers to current batch and head
    prob_start = prob_ptr + batch_id * (heads * seqlen * seqlen) + head_id * (seqlen * seqlen)
    v_start = v_ptr + batch_id * (heads * seqlen * dim) + head_id * (seqlen * dim)
    out_start = out_ptr + batch_id * (heads * seqlen * dim) + head_id * (seqlen * dim)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Iterate over sequence length in steps of BLOCK_N
    for n in range(window_start, window_end, BLOCK_N):
        # Bounds checking
        n_size = tl.minimum(BLOCK_N, window_end - n)
        
        # Load probabilities and values
        p = tl.load(prob_start + start_m * seqlen + n + offs_m[:, None] * seqlen + offs_n[None, :n_size])
        v = tl.load(v_start + n * dim + offs_n[:n_size, None] * dim + offs_d[None, :])
        
        # Compute attention
        acc += tl.dot(p, v)
    
    # Store result
    for d in range(0, dim, BLOCK_DMODEL):
        d_size = tl.minimum(BLOCK_DMODEL, dim - d)
        tl.store(out_start + start_m * dim + offs_m[:, None] * dim + offs_d[None, :d_size], acc[:, :d_size])

def token_att_fwd2(prob, v, window_size=None):
    """
    Forward pass for token attention.
    
    Args:
        prob: attention probabilities (batch, heads, seqlen, seqlen)
        v: values (batch, heads, seqlen, dim)
        window_size: size of the attention window (optional)
    
    Returns:
        out: output tensor (batch, heads, seqlen, dim)
    """
    batch, heads, seqlen, _ = prob.shape
    dim = v.shape[-1]
    
    # Set default window size if not provided
    if window_size is None:
        window_size = seqlen
    
    # Allocate output
    out = torch.empty_like(v)
    
    # Block sizes (tuned for efficiency)
    BLOCK_M = 16
    BLOCK_N = 32
    BLOCK_DMODEL = 32
    
    # Launch kernel
    grid = (batch * heads,)
    _fwd_kernel_token_att2[grid](
        prob, v, out,
        batch, heads, seqlen, dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        window_size=window_size,
    )
    
    return out
