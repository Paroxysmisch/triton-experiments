import torch
import triton
import triton.language as tl

@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, h_ptr, o_ptr,
    # Matrix dimensions
    seqlen, dim, # Original dimensions
    BT: tl.constexpr, # Chunk size for sequence length
    BK: tl.constexpr, # Block size for key dimension
    BV: tl.constexpr, # Block size for value dimension
    BLOCK_M: tl.constexpr, # Block size for M dimension
    BLOCK_N: tl.constexpr, # Block size for N dimension
    BLOCK_DMODEL: tl.constexpr, # Block size for model dimension
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate current block indices
    block_start = pid * BT
    
    # Create block pointers
    q_block_ptr = tl.make_block_ptr(
        q_ptr, (seqlen, dim),
        (dim, 1),
        (block_start, 0),
        (BT, BK),
        (1, 0)
    )
    
    k_block_ptr = tl.make_block_ptr(
        k_ptr, (seqlen, dim),
        (dim, 1),
        (block_start, 0),
        (BT, BK),
        (1, 0)
    )
    
    v_block_ptr = tl.make_block_ptr(
        v_ptr, (seqlen, dim),
        (dim, 1),
        (block_start, 0),
        (BT, BV),
        (1, 0)
    )
    
    h_block_ptr = tl.make_block_ptr(
        h_ptr, (seqlen, dim),
        (dim, 1),
        (block_start, 0),
        (BT, BK),
        (1, 0)
    )
    
    # Load blocks
    q = tl.load(q_block_ptr)
    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)
    h = tl.load(h_block_ptr)
    
    # Compute attention scores
    scores = tl.dot(q, tl.trans(k))
    
    # Apply scaling
    scale = 1.0 / tl.sqrt(float(dim))
    scores = scores * scale
    
    # Apply exponential and gating
    exp_scores = tl.exp(scores)
    gated_scores = exp_scores * h
    
    # Compute output
    o = tl.dot(gated_scores, v)
    
    # Store results
    o_block_ptr = tl.make_block_ptr(
        o_ptr, (seqlen, dim),
        (dim, 1),
        (block_start, 0),
        (BT, BV),
        (1, 0)
    )
    tl.store(o_block_ptr, o)

def chunk_fwd_o_fn(q, k, v, h):
    """
    Wrapper function for the chunked GLA forward pass
    Args:
        q: Query tensor of shape (seqlen, dim)
        k: Key tensor of shape (seqlen, dim)
        v: Value tensor of shape (seqlen, dim)
        h: Auxiliary tensor of shape (seqlen, dim)
    Returns:
        o: Output tensor of shape (seqlen, dim)
    """
    seqlen, dim = q.shape
    
    # Initialize output
    o = torch.empty_like(v)
    
    # Configure block sizes
    BT = 32  # Chunk size for sequence length
    BK = 32  # Block size for key dimension
    BV = 32  # Block size for value dimension
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_DMODEL = 32
    
    # Calculate grid size
    grid = (triton.cdiv(seqlen, BT),)
    
    # Launch kernel
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, o,
        seqlen, dim,
        BT=BT, BK=BK, BV=BV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL
    )
    
    return o
