import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, o_ptr,
    # Matrix dimensions
    batch, heads, seq_len, dim,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    # Other parameters
    scale,
    window_size: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Block indices
    block_m = pid // (seq_len // BLOCK_M)
    block_n = pid % (seq_len // BLOCK_N)

    # Initialize offsets
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Initialize pointers
    q_block_ptr = q_ptr + offs_m[:, None] * dim + offs_d[None, :]
    k_block_ptr = k_ptr + offs_n[:, None] * dim + offs_d[None, :]
    
    # Load query and key blocks
    q = tl.load(q_block_ptr, mask=offs_m[:, None] < seq_len, other=0.0)
    k = tl.load(k_block_ptr, mask=offs_n[:, None] < seq_len, other=0.0)
    
    # Compute attention scores
    scores = tl.dot(q, k.transpose())
    scores = scores * scale
    
    # Apply window mask if needed
    if window_size > 0:
        window_mask = tl.abs(offs_m[:, None] - offs_n[None, :]) < window_size
        scores = tl.where(window_mask, scores, float("-inf"))
    
    # Compute softmax
    scores_max = tl.max(scores, 1)
    scores = scores - scores_max[:, None]
    scores_exp = tl.exp(scores)
    scores_sum = tl.sum(scores_exp, 1)
    scores = scores_exp / scores_sum[:, None]
    
    # Load values and compute output
    v_block_ptr = v_ptr + offs_n[:, None] * dim + offs_d[None, :]
    v = tl.load(v_block_ptr, mask=offs_n[:, None] < seq_len, other=0.0)
    
    output = tl.dot(scores, v)
    
    # Store output
    o_block_ptr = o_ptr + offs_m[:, None] * dim + offs_d[None, :]
    tl.store(o_block_ptr, output, mask=offs_m[:, None] < seq_len)

# Wrapper function
def attention_forward(q, k, v, window_size=-1):
    batch, heads, seq_len, dim = q.shape
    scale = 1.0 / (dim ** 0.5)
    
    # Allocate output
    output = torch.empty_like(q)
    
    # Configure block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_D = dim
    
    # Grid size
    grid = (seq_len * seq_len) // (BLOCK_M * BLOCK_N)
    
    # Launch kernel
    _attn_fwd_kernel[grid](
        q, k, v, output,
        batch, heads, seq_len, dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        scale=scale,
        window_size=window_size,
    )
    
    return output
