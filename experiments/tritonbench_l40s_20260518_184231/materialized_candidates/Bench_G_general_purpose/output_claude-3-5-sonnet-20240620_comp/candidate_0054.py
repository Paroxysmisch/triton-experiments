import triton
import triton.language as tl
import torch

@triton.jit
def chunk_gated_abc_fwd_kernel_cum(
    o_ptr, s_ptr,
    stride_os_t, stride_os_s,
    T, S,
    BT: tl.constexpr, BS: tl.constexpr,
):
    # Program ID
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    # Block pointers
    block_s = pid_s * BS
    block_t = pid_t * BT
    
    # Create mask for valid elements
    offs_t = block_t + tl.arange(0, BT)
    offs_s = block_s + tl.arange(0, BS)
    m_s = offs_s < S
    m_t = offs_t < T
    
    # Compute input/output pointers
    s_ptrs = s_ptr + offs_t[:, None] * stride_os_t + offs_s[None, :] * stride_os_s
    o_ptrs = o_ptr + offs_t[:, None] * stride_os_t + offs_s[None, :] * stride_os_s
    
    # Load input data with mask
    x = tl.load(s_ptrs, mask=m_t[:, None] & m_s[None, :], other=0.0)
    
    # Compute cumulative sum along time dimension
    x = tl.cumsum(x, axis=0)
    
    # Store result
    tl.store(o_ptrs, x, mask=m_t[:, None] & m_s[None, :])

@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    h_ptr, k_ptr, v_ptr, g_ptr,
    h0_ptr, ht_ptr,
    stride_h_t, stride_h_v,
    stride_k_t, stride_k_k,
    stride_v_t, stride_v_v,
    stride_g_t,
    T, K, V,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
):
    # Program ID
    pid_t = tl.program_id(0)
    pid_v = tl.program_id(1)
    
    # Block pointers
    block_t = pid_t * BT
    block_v = pid_v * BV
    
    # Create masks
    offs_t = block_t + tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    offs_v = block_v + tl.arange(0, BV)
    
    m_t = offs_t < T
    m_v = offs_v < V
    
    # Initialize accumulators
    b_h = tl.zeros([BT, BV], dtype=tl.float32)
    
    # Load initial state if needed
    if USE_INITIAL_STATE:
        h0_ptrs = h0_ptr + offs_v[None, :] * stride_h_v
        b_h = tl.where(m_v[None, :], tl.load(h0_ptrs), b_h)
    
    # Main loop over K dimension
    for k in range(0, K, BK):
        k_ptrs = k_ptr + offs_t[:, None] * stride_k_t + (k + offs_k)[None, :] * stride_k_k
        v_ptrs = v_ptr + (k + offs_k)[:, None] * stride_v_t + offs_v[None, :] * stride_v_v
        g_ptrs = g_ptr + offs_t * stride_g_t + k
        
        # Load and compute
        b_k = tl.load(k_ptrs, mask=m_t[:, None], other=0.0)
        b_v = tl.load(v_ptrs, mask=m_v[None, :], other=0.0)
        b_g = tl.load(g_ptrs, mask=m_t, other=0.0)
        
        # Update hidden state
        b_h = b_h + tl.dot(b_k, b_v) * b_g[:, None]
    
    # Store results
    h_ptrs = h_ptr + offs_t[:, None] * stride_h_t + offs_v[None, :] * stride_h_v
    tl.store(h_ptrs, b_h, mask=m_t[:, None] & m_v[None, :])
    
    # Store final state if needed
    if USE_FINAL_STATE:
        ht_ptrs = ht_ptr + offs_v * stride_h_v
        last_t = tl.max(tl.where(m_t, offs_t, -1))
        tl.store(ht_ptrs, b_h[last_t], mask=m_v)

def fwd_pre(g):
    T, S = g.shape
    BLOCK_T = 32
    BLOCK_S = 32
    
    # Prepare output tensor
    o = torch.empty_like(g)
    
    # Launch kernel
    grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(S, BLOCK_S))
    chunk_gated_abc_fwd_kernel_cum[grid](
        o, g,
        g.stride(0), g.stride(1),
        T, S,
        BLOCK_T, BLOCK_S
    )
    return o

def fwd_inner(k, v, g, h0=None, return_state=False):
    T, K = k.shape
    _, V = v.shape
    
    # Define block sizes
    BLOCK_T = 32
    BLOCK_K = 32
    BLOCK_V = 32
    
    # Prepare output tensor
    h = torch.empty((T, V), device=k.device, dtype=k.dtype)
    ht = torch.empty((V,), device=k.device, dtype=k.dtype) if return_state else None
    
    # Launch kernel
    grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(V, BLOCK_V))
    chunk_gated_abc_fwd_kernel_h[grid](
        h, k, v, g,
        h0 if h0 is not None else h,  # Use h as dummy if h0 is None
        ht if return_state else h,    # Use h as dummy if ht not needed
        h.stride(0), h.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        g.stride(0),
        T, K, V,
        BLOCK_T, BLOCK_K, BLOCK_V,
        h0 is not None,
        return_state
    )
    
    if return_state:
        return h, ht
    return h
