import triton
import triton.language as tl

# Kernel for cumulative matrix computation
@triton.jit
def chunk_gated_abc_fwd_kernel_cum(
    s_ptr, o_ptr, 
    stride_s_b, stride_s_t, stride_s_s, 
    stride_o_b, stride_o_t, stride_o_s, 
    T, S, BT, BS, 
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_S: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // (T // BLOCK_SIZE_T)
    tid = pid % (T // BLOCK_SIZE_T)
    
    # Block bounds
    block_start_t = tid * BLOCK_SIZE_T
    block_start_s = bid * BLOCK_SIZE_S
    
    # Initialize pointers
    s_block_ptr = s_ptr + bid * stride_s_b + block_start_t * stride_s_t + block_start_s * stride_s_s
    o_block_ptr = o_ptr + bid * stride_o_b + block_start_t * stride_o_t + block_start_s * stride_o_s
    
    # Load data
    s = tl.load(s_block_ptr, mask=block_start_s + tl.arange(0, BLOCK_SIZE_S) < S, other=0.0)
    
    # Compute mask for upper triangular part
    m_s = tl.arange(0, BLOCK_SIZE_S) >= block_start_t + tl.arange(0, BLOCK_SIZE_T)
    
    # Compute cumulative sum
    for i in range(BLOCK_SIZE_T):
        s = tl.where(m_s[i], s, 0.0)
        tl.atomic_add(o_block_ptr + i * stride_o_t, s)

# Kernel for gated accumulation
@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k_ptr, v_ptr, g_ptr, h_ptr, h0_ptr, ht_ptr, 
    stride_k_b, stride_k_t, stride_k_s, 
    stride_v_b, stride_v_t, stride_v_s, 
    stride_g_b, stride_g_t, stride_g_s, 
    stride_h_b, stride_h_t, stride_h_s, 
    stride_h0_b, stride_h0_t, stride_h0_s, 
    stride_ht_b, stride_ht_t, stride_ht_s, 
    T, S, BT, BS, 
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_S: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // (T // BLOCK_SIZE_T)
    tid = pid % (T // BLOCK_SIZE_T)
    
    # Block bounds
    block_start_t = tid * BLOCK_SIZE_T
    block_start_s = bid * BLOCK_SIZE_S
    
    # Initialize pointers
    k_block_ptr = k_ptr + bid * stride_k_b + block_start_t * stride_k_t + block_start_s * stride_k_s
    v_block_ptr = v_ptr + bid * stride_v_b + block_start_t * stride_v_t + block_start_s * stride_v_s
    g_block_ptr = g_ptr + bid * stride_g_b + block_start_t * stride_g_t + block_start_s * stride_g_s
    h_block_ptr = h_ptr + bid * stride_h_b + block_start_t * stride_h_t + block_start_s * stride_h_s
    
    # Load initial state if provided
    if h0_ptr is not None:
        h0_block_ptr = h0_ptr + bid * stride_h0_b + block_start_t * stride_h0_t + block_start_s * stride_h0_s
        h0 = tl.load(h0_block_ptr, mask=block_start_s + tl.arange(0, BLOCK_SIZE_S) < S, other=0.0)
        h = h0
    else:
        h = tl.zeros((BLOCK_SIZE_S,), dtype=tl.float32)
    
    # Load data
    k = tl.load(k_block_ptr, mask=block_start_s + tl.arange(0, BLOCK_SIZE_S) < S, other=0.0)
    v = tl.load(v_block_ptr, mask=block_start_s + tl.arange(0, BLOCK_SIZE_S) < S, other=0.0)
    g = tl.load(g_block_ptr, mask=block_start_s + tl.arange(0, BLOCK_SIZE_S) < S, other=0.0)
    
    # Compute gated accumulation
    for i in range(BLOCK_SIZE_T):
        h = h + g * k * v
        tl.atomic_add(h_block_ptr + i * stride_h_t, h)
    
    # Store final state if provided
    if ht_ptr is not None:
        ht_block_ptr = ht_ptr + bid * stride_ht_b + block_start_t * stride_ht_t + block_start_s * stride_ht_s
        tl.store(ht_block_ptr, h, mask=block_start_s + tl.arange(0, BLOCK_SIZE_S) < S)

### Wrapper Functions

def fwd_pre(s, o, T, S, BT, BS, BLOCK_SIZE_T, BLOCK_SIZE_S):
    grid = (T * S // (BLOCK_SIZE_T * BLOCK_SIZE_S),)
    chunk_gated_abc_fwd_kernel_cum[grid](
        s, o, 
        s.stride(0), s.stride(1), s.stride(2), 
        o.stride(0), o.stride(1), o.stride(2), 
        T, S, BT, BS, 
        BLOCK_SIZE_T, BLOCK_SIZE_S
    )

def fwd_inner(k, v, g, h, h0=None, ht=None, T, S, BT, BS, BLOCK_SIZE_T, BLOCK_SIZE_S):
    grid = (T * S // (BLOCK_SIZE_T * BLOCK_SIZE_S),)
    chunk_gated_abc_fwd_kernel_h[grid](
        k, v, g, h, h0, ht, 
        k.stride(0), k.stride(1), k.stride(2), 
        v.stride(0), v.stride(1), v.stride(2), 
        g.stride(0), g.stride(1), g.stride(2), 
        h.stride(0), h.stride(1), h.stride(2), 
        h0.stride(0) if h0 is not None else 0, h0.stride(1) if h0 is not None else 0, h0.stride(2) if h0 is not None else 0, 
        ht.stride(0) if ht is not None else 0, ht.stride(1) if ht is not None else 0, ht.stride(2) if ht is not None else 0, 
        T, S, BT, BS, 
        BLOCK_SIZE_T, BLOCK_SIZE_S
    )
