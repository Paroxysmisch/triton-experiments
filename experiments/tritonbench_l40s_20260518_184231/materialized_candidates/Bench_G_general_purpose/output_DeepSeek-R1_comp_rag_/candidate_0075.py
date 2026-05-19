import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BT': 64, 'BK': 64, 'BV': 64}, num_warps=4),
        triton.Config({'BT': 128, 'BK': 64, 'BV': 64}, num_warps=4),
        triton.Config({'BT': 64, 'BK': 128, 'BV': 64}, num_warps=4),
        triton.Config({'BT': 128, 'BK': 128, 'BV': 128}, num_warps=8),
    ],
    key=['T', 'H']
)
@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    # Tensors
    q_ptr, k_ptr, v_ptr, h_ptr, g_ptr, o_ptr,
    # Strides
    s_k_h: tl.constexpr, s_k_t: tl.constexpr,
    s_q_h: tl.constexpr, s_q_t: tl.constexpr,
    s_v_b: tl.constexpr, s_v_h: tl.constexpr, s_v_t: tl.constexpr,
    s_h_h: tl.constexpr, s_h_t: tl.constexpr,
    s_g_h: tl.constexpr, s_g_t: tl.constexpr,
    s_o_b: tl.constexpr, s_o_h: tl.constexpr, s_o_t: tl.constexpr,
    # Params
    scale: tl.constexpr,
    T: tl.constexpr, H: tl.constexpr,
    # Chunk sizes
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    # Meta-parameters
    USE_MASK: tl.constexpr,
):
    # Parallel computation indices
    bid = tl.program_id(0)
    tid = tl.program_id(1)
    
    # Decompose indices
    num_heads = H
    num_chunks = tl.cdiv(T, BT)
    batch = bid // (num_heads * num_chunks)
    head = (bid // num_chunks) % num_heads
    chunk = bid % num_chunks
    
    # Offsets
    t_start = chunk * BT
    g_off = head * s_g_h + t_start * s_g_t
    h_off = batch * s_h_h + head * s_h_t + t_start
    
    # Block pointers
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + batch * s_q_h + head * s_q_t,
        shape=(T, BK), strides=(s_q_t, 1),
        offsets=(t_start, 0), block_shape=(BT, BK), order=(1, 0)
    )
    
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + batch * s_k_h + head * s_k_t,
        shape=(BK, T), strides=(1, s_k_t),
        offsets=(0, t_start), block_shape=(BK, BT), order=(0, 1)
    )
    
    # Initialize accumulators
    b_o = tl.zeros((BV, BT), dtype=tl.float32)
    b_s = tl.zeros((BV, BT), dtype=tl.float32)
    
    # Main computation loop
    for _ in range(0, tl.cdiv(T, BT)):
        q = tl.load(q_block_ptr, boundary_check=(0, 1))
        k = tl.load(k_block_ptr, boundary_check=(1, 0))
        
        # Compute attention scores
        s = tl.dot(q, k) * scale
        if USE_MASK:
            s = tl.where(t_start >= tid, s, float('-inf'))
        
        # Load previous state
        h = tl.load(h_ptr + h_off, mask=h_off < T, other=0.0)
        g = tl.load(g_ptr + g_off, mask=g_off < T, other=0.0)
        
        # Update accumulators
        e_s = tl.exp(s - tl.max(s, axis=0))
        b_o += e_s * (h * g)[:, None]
        b_s += e_s * g[:, None]
        
        # Update pointers
        q_block_ptr = tl.advance(q_block_ptr, (0, BK))
        k_block_ptr = tl.advance(k_block_ptr, (BK, 0))
        h_off += BT
        g_off += BT * s_g_t
    
    # Store results
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + batch * s_o_b + head * s_o_h,
        shape=(BT, BV), strides=(s_o_t, 1),
        offsets=(t_start, 0), block_shape=(BT, BV), order=(1, 0)
    )
    tl.store(o_block_ptr, b_o / b_s, boundary_check=(0, 1))

def chunk_fwd_o_fn(q, k, v, h, g, scale=1.0):
    # Validate inputs
    assert q.dim() == 4, "q must be 4D (batch, heads, time, dim)"
    B, H, T, D = q.shape
    
    # Allocate output
    o = torch.empty_like(v)
    
    # Determine chunk sizes
    BT = triton.next_power_of_2(T // 4) if T > 64 else 32
    BK = min(triton.next_power_of_2(D), 128)
    BV = min(v.size(-1), 128)
    
    # Grid configuration
    grid = (B * H * triton.cdiv(T, BT),)
    
    # Launch kernel
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, g, o,
        s_k_h=k.stride(0), s_k_t=k.stride(2),
        s_q_h=q.stride(0), s_q_t=q.stride(2),
        s_v_b=v.stride(0), s_v_h=v.stride(1), s_v_t=v.stride(2),
        s_h_h=h.stride(0), s_h_t=h.stride(1),
        s_g_h=g.stride(0), s_g_t=g.stride(1),
        s_o_b=o.stride(0), s_o_h=o.stride(1), s_o_t=o.stride(2),
        scale=scale,
        T=T, H=H,
        BT=BT, BK=BK, BV=BV,
        USE_MASK=not h.is_contiguous()
    )
    return o
