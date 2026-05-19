import triton
import triton.language as tl

@triton.jit
def chunk_gated_abc_fwd_kernel_cum(
    s_ptr, o_ptr,
    stride_s0, stride_s1,
    stride_o0, stride_o1,
    T, S,
    BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr
):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)

    t_start = pid_t * BLOCK_T
    s_start = pid_s * BLOCK_S

    # create offsets
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    s_offsets = s_start + tl.arange(0, BLOCK_S)

    # broadcast for indexing
    t_offsets = tl.reshape(t_offsets, [BLOCK_T, 1])
    s_offsets = tl.reshape(s_offsets, [1, BLOCK_S])

    # pointers
    s_ptrs = s_ptr + (t_offsets * stride_s0) + (s_offsets * stride_s1)
    o_ptrs = o_ptr + (t_offsets * stride_o0) + (s_offsets * stride_o1)

    # mask
    mask_t = t_offsets < T
    mask_s = s_offsets < S
    mask = mask_t & mask_s

    # load
    s_vals = tl.where(mask, tl.load(s_ptrs, mask=mask, other=0.0), 0.0)

    # cumulative operation (example: partial sum over columns)
    # This example does a row-wise cumulative sum across the dimension S
    # Each block processes a slice, so the partial sums can be integrated across blocks if needed
    cumsum_vals = tl.zeros([BLOCK_T, BLOCK_S], dtype=tl.float32)
    running_sum = 0.0
    for i in range(BLOCK_S):
        val = s_vals[:, i]
        running_sum = running_sum + val
        cumsum_vals[:, i] = running_sum

    # store
    tl.store(o_ptrs, cumsum_vals, mask=mask)


@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k_ptr, v_ptr, g_ptr, h0_ptr, ht_ptr, h_ptr,
    stride_k0, stride_k1,
    stride_v0, stride_v1,
    stride_g0, stride_g1,
    stride_h0, stride_ht,
    stride_h0_out, stride_h1_out,
    T, K, V,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
    HAS_STATE: tl.constexpr
):
    pid_t = tl.program_id(0)

    t_start = pid_t * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    # pointers for k, v, g
    k_ptrs = k_ptr + t_offsets[:, None] * stride_k0 + tl.arange(0, BLOCK_K)[None, :] * stride_k1
    v_ptrs = v_ptr + t_offsets[:, None] * stride_v0 + tl.arange(0, BLOCK_V)[None, :] * stride_v1
    g_ptrs = g_ptr + t_offsets[:, None] * stride_g0 + tl.arange(0, BLOCK_K)[None, :] * stride_g1

    # load input
    k_vals = tl.where(mask_t[:, None], tl.load(k_ptrs, mask=mask_t[:, None], other=0.0), 0.0)
    v_vals = tl.where(mask_t[:, None], tl.load(v_ptrs, mask=mask_t[:, None], other=0.0), 0.0)
    g_vals = tl.where(mask_t[:, None], tl.load(g_ptrs, mask=mask_t[:, None], other=0.0), 0.0)

    # possibly load initial state
    h0_vals = tl.zeros([BLOCK_T, BLOCK_V], dtype=tl.float32)
    if HAS_STATE:
        h0_vals = tl.where(mask_t[:, None],
                           tl.load(h0_ptr + t_offsets[:, None] * stride_h0 + tl.arange(0, BLOCK_V)[None, :] * stride_h0_out,
                                   mask=mask_t[:, None],
                                   other=0.0),
                           0.0)

    # gated accumulation example
    # For each row t, compute h = h0 + (k * g) + (v * (1 - g)) in some fashion
    # This is just an example logic placeholder
    b_h = h0_vals
    for i in range(BLOCK_K):
        gate = g_vals[:, i]
        b_h = b_h + k_vals[:, i] * gate[:, None] + v_vals[:, i] * (1.0 - gate)[:, None]

    # store final or interim states
    if HAS_STATE:
        # store final state per chunk
        final_ptrs = ht_ptr + t_offsets[:, None] * stride_ht
        tl.store(final_ptrs, b_h, mask=mask_t[:, None])

    # write to output
    out_ptrs = h_ptr + t_offsets[:, None] * stride_h1_out
    tl.store(out_ptrs, b_h, mask=mask_t[:, None])


def fwd_pre(s, o, T, S, BLOCK_T=128, BLOCK_S=128):
    grid = lambda meta: ( (T + BLOCK_T - 1) // BLOCK_T,
                          (S + BLOCK_S - 1) // BLOCK_S )
    chunk_gated_abc_fwd_kernel_cum[grid](
        s, o,
        s.stride(0), s.stride(1),
        o.stride(0), o.stride(1),
        T, S,
        BLOCK_T=BLOCK_T, BLOCK_S=BLOCK_S
    )


def fwd_inner(k, v, g, h0, ht, h, T, K, V, has_state=True, BLOCK_T=128, BLOCK_K=32, BLOCK_V=32):
    grid = lambda meta: ( (T + BLOCK_T - 1) // BLOCK_T, )
    chunk_gated_abc_fwd_kernel_h[grid](
        k, v, g, h0, ht, h,
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        g.stride(0), g.stride(1),
        h0.stride(0) if has_state else 0, ht.stride(0) if has_state else 0,
        h0.stride(1) if has_state else 0, h.stride(1),
        T, K, V,
        BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
        HAS_STATE=has_state
    )
