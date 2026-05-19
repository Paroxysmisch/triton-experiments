import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, h_ptr, g_ptr, o_ptr,
    # Matrix strides
    s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t,
    # Scaling factor
    scale,
    # Problem size
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize pointers to blocks
    q_block_ptr = q_ptr + offs_m[:, None] * s_k_h + offs_k[None, :] * s_k_t
    k_block_ptr = k_ptr + offs_k[:, None] * s_k_h + offs_n[None, :] * s_k_t
    v_block_ptr = v_ptr + offs_n[:, None] * s_v_h + offs_k[None, :] * s_v_t
    h_block_ptr = h_ptr + offs_m[:, None] * s_h_h + offs_n[None, :] * s_h_t

    # Initialize accumulators
    b_o = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    b_s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Load blocks
    q = tl.load(q_block_ptr, mask=offs_m[:, None] < BK, other=0.0)
    k = tl.load(k_block_ptr, mask=offs_k[:, None] < BT, other=0.0)
    v = tl.load(v_block_ptr, mask=offs_n[:, None] < BV, other=0.0)
    h = tl.load(h_block_ptr, mask=offs_m[:, None] < BK, other=0.0)

    # Compute attention scores
    scores = tl.dot(q, k) * scale
    
    # Apply softmax
    scores = scores - tl.max(scores, axis=1)[:, None]
    exp_scores = tl.exp(scores)
    
    # Compute mask for valid positions
    m_s = offs_m[:, None] < BK
    
    # Apply mask and compute weighted sum
    exp_scores = tl.where(m_s, exp_scores, 0.0)
    b_s += tl.sum(exp_scores, axis=1)[:, None]
    
    # Compute output
    b_o = tl.dot(exp_scores, v)
    b_o = b_o * h
    
    # Store results
    o_block_ptr = o_ptr + offs_m[:, None] * s_k_h + offs_n[None, :] * s_k_t
    tl.store(o_block_ptr, b_o, mask=offs_m[:, None] < BK)

def chunk_fwd_o_fn(q, k, v, h, g, scale):
    # Extract dimensions
    BK = q.shape[0]  # Batch size * Key dimension
    BT = k.shape[1]  # Target sequence length
    BV = v.shape[1]  # Value dimension
    
    # Allocate output
    o = torch.empty_like(q)
    
    # Compute grid dimensions
    grid = (triton.cdiv(BK, 128), triton.cdiv(BT, 128))
    
    # Launch kernel
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, g, o,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        scale,
        BT, BK, BV,
        128, 128, 32
    )
    
    return o
