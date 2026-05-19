import triton
import triton.language as tl
import torch

@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q, k, v, h, g, o,
    s_q_h, s_q_t, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, s_g_h, s_g_t, s_o_h, s_o_t,
    scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    H: tl.constexpr, T: tl.constexpr, K: tl.constexpr
):
    # Compute program ID
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)

    # Compute offsets
    off_h = pid_h * BT
    off_t = pid_t * BK
    off_v = pid_t * BV

    # Load query block
    q_ptr = q + off_h * s_q_h + tl.arange(0, BT) * s_q_t
    b_q = tl.load(q_ptr, mask=tl.arange(0, BT) < T, other=0.0)

    # Initialize accumulators
    b_o = tl.zeros((BT, BV), dtype=tl.float32)
    b_s = tl.zeros((BT,), dtype=tl.float32)

    # Loop over key chunks
    for _ in range(0, K, BK):
        # Load key and value blocks
        k_ptr = k + off_t * s_k_h + tl.arange(0, BK) * s_k_t
        v_ptr = v + off_v * s_v_h + tl.arange(0, BV) * s_v_t
        b_k = tl.load(k_ptr, mask=tl.arange(0, BK) < K, other=0.0)
        b_v = tl.load(v_ptr, mask=tl.arange(0, BV) < K, other=0.0)

        # Compute attention scores
        b_s_partial = tl.dot(b_q, b_k) * scale

        # Apply softmax
        b_s_partial = tl.exp(b_s_partial - tl.max(b_s_partial, axis=1)[:, None])
        b_s += tl.sum(b_s_partial, axis=1)

        # Update output accumulator
        b_o += tl.dot(b_s_partial, b_v)

        # Move to next key chunk
        off_t += BK
        off_v += BV

    # Load h and g blocks
    h_ptr = h + off_h * s_h_h + tl.arange(0, BT) * s_h_t
    g_ptr = g + off_h * s_g_h + tl.arange(0, BT) * s_g_t
    b_h = tl.load(h_ptr, mask=tl.arange(0, BT) < T, other=0.0)
    b_g = tl.load(g_ptr, mask=tl.arange(0, BT) < T, other=0.0)

    # Compute final output
    b_o = b_o / b_s[:, None]
    b_o = b_h * b_o + b_g * b_q

    # Store output
    o_ptr = o + off_h * s_o_h + tl.arange(0, BT) * s_o_t
    tl.store(o_ptr, b_o, mask=tl.arange(0, BT) < T)

@triton.autotune(
    configs=[
        triton.Config({'BT': 32, 'BK': 32, 'BV': 32}, num_stages=3, num_warps=4),
        triton.Config({'BT': 64, 'BK': 32, 'BV': 32}, num_stages=3, num_warps=8),
        triton.Config({'BT': 128, 'BK': 32, 'BV': 32}, num_stages=3, num_warps=8),
    ],
    key=['H', 'T', 'K']
)
@triton.jit
def chunk_fwd_o_fn(
    q, k, v, h, g, o,
    s_q_h, s_q_t, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, s_g_h, s_g_t, s_o_h, s_o_t,
    H: tl.constexpr, T: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    scale = 1.0 / (K ** 0.5)
    
    # Launch kernel
    grid = (triton.cdiv(T, BT), H)
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, g, o,
        s_q_h, s_q_t, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, s_g_h, s_g_t, s_o_h, s_o_t,
        scale,
        BT, BK, BV,
        H, T, K
    )

def chunk_gla_forward(q, k, v, h, g):
    # Ensure inputs are on GPU and have correct dtype
    assert q.is_cuda and k.is_cuda and v.is_cuda and h.is_cuda and g.is_cuda
    assert q.dtype == k.dtype == v.dtype == h.dtype == g.dtype == torch.float32

    # Extract dimensions
    H, T, K = q.shape
    _, _, V = v.shape

    # Prepare output tensor
    o = torch.empty_like(q)

    # Compute strides
    s_q_h, s_q_t = q.stride()
    s_k_h, s_k_t = k.stride()
    s_v_h, s_v_t = v.stride()
    s_h_h, s_h_t = h.stride()
    s_g_h, s_g_t = g.stride()
    s_o_h, s_o_t = o.stride()

    # Determine optimal block sizes
    BT = min(T, 128)
    BK = min(K, 32)
    BV = min(V, 32)

    # Launch Triton kernel
    chunk_fwd_o_fn[(H, T, K)](
        q, k, v, h, g, o,
        s_q_h, s_q_t, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, s_g_h, s_g_t, s_o_h, s_o_t,
        H, T, K,
        BT=BT, BK=BK, BV=BV
    )

    return o
