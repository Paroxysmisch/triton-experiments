import triton
import triton.language as tl

@triton.jit
def chunk_gated_abc_fwd_kernel_cum(
    s_ptr, o_ptr,
    T, S,
    s_tstride, s_sstride,
    o_tstride, o_sstride,
    BT: tl.constexpr, BS: tl.constexpr
):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    off_t = pid_t * BT + tl.arange(0, BT)
    off_s = pid_s * BS + tl.arange(0, BS)
    
    mask_t = off_t < T
    mask_s = off_s < S
    mask = mask_t[:, None] & mask_s[None, :]
    
    s_ptrs = s_ptr + off_t[:, None] * s_tstride + off_s[None, :] * s_sstride
    o_ptrs = o_ptr + off_t[:, None] * o_tstride + off_s[None, :] * o_sstride
    
    s_vals = tl.load(s_ptrs, mask=mask, other=0.0)
    
    # Compute cumulative sum along the S dimension within the block
    cumsum = tl.associative_scan(s_vals, axis=1, combine_fn=lambda a, b: a + b)
    
    tl.store(o_ptrs, cumsum, mask=mask)

@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k_ptr, v_ptr, g_ptr, h0_ptr, ht_ptr, h_ptr,
    T, K, V,
    k_tstride, k_kstride, k_vstride,
    v_tstride, v_kstride, v_vstride,
    g_tstride, g_kstride, g_vstride,
    h_tstride, h_kstride, h_vstride,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    has_initial: tl.constexpr, save_final: tl.constexpr
):
    pid_t = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_v = tl.program_id(2)
    
    off_t = pid_t * BT + tl.arange(0, BT)
    off_k = pid_k * BK + tl.arange(0, BK)
    off_v = pid_v * BV + tl.arange(0, BV)
    
    mask_t = off_t < T
    mask_k = off_k < K
    mask_v = off_v < V
    full_mask = mask_t[:, None, None] & mask_k[None, :, None] & mask_v[None, None, :]
    
    # Load initial state if provided
    if has_initial:
        h0_ptrs = h0_ptr + off_k[:, None] * h_kstride + off_v[None, :] * h_vstride
        b_h = tl.load(h0_ptrs, mask=mask_k[:, None] & mask_v[None, :], other=0.0)
    else:
        b_h = tl.zeros((BK, BV), dtype=tl.float32)
    
    for t_step in range(BT):
        t = off_t + t_step
        current_mask = (t < T) & full_mask
        
        # Load current k, v, g
        k_ptrs = k_ptr + t[:, None, None] * k_tstride + off_k[None, :, None] * k_kstride + off_v[None, None, :] * k_vstride
        v_ptrs = v_ptr + t[:, None, None] * v_tstride + off_k[None, :, None] * v_kstride + off_v[None, None, :] * v_vstride
        g_ptrs = g_ptr + t[:, None, None] * g_tstride + off_k[None, :, None] * g_kstride + off_v[None, None, :] * g_vstride
        
        k = tl.load(k_ptrs, mask=current_mask, other=0.0)
        v = tl.load(v_ptrs, mask=current_mask, other=0.0)
        g = tl.load(g_ptrs, mask=current_mask, other=0.0)
        
        # Gated update
        new_h = g * (k + v) + (1 - g) * b_h
        
        # Store to output
        h_ptrs = h_ptr + t[:, None, None] * h_tstride + off_k[None, :, None] * h_kstride + off_v[None, None, :] * h_vstride
        tl.store(h_ptrs, new_h, mask=current_mask)
        
        b_h = new_h
    
    if save_final:
        ht_ptrs = ht_ptr + off_k[:, None] * h_kstride + off_v[None, :] * h_vstride
        tl.store(ht_ptrs, b_h, mask=mask_k[:, None] & mask_v[None, :])

def fwd_pre(g):
    T, S = g.shape
    BT, BS = 128, 64  # Tune based on hardware
    o = torch.empty_like(g)
    grid = (triton.cdiv(T, BT), triton.cdiv(S, BS))
    chunk_gated_abc_fwd_kernel_cum[grid](
        g, o,
        T, S,
        g.stride(0), g.stride(1),
        o.stride(0), o.stride(1),
        BT=BT, BS=BS
    )
    return o

def fwd_inner(k, v, g, h0=None, ht=None):
    T, K, V = k.shape[0], k.shape[1], v.shape[2]
    BT, BK, BV = 32, 64, 64  # Tune based on hardware
    has_initial = h0 is not None
    save_final = ht is not None
    
    h = torch.empty((T, K, V), device=k.device, dtype=k.dtype)
    grid = (
        triton.cdiv(T, BT),
        triton.cdiv(K, BK),
        triton.cdiv(V, BV)
    )
    
    chunk_gated_abc_fwd_kernel_h[grid](
        k, v, g, h0, ht, h,
        T, K, V,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        g.stride(0), g.stride(1), g.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        BT=BT, BK=BK, BV=BV,
        has_initial=has_initial, save_final=save_final
    )
    return h
