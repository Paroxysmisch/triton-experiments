import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BD': 32}, num_warps=1),
        triton.Config({'BD': 32}, num_warps=2),
        triton.Config({'BD': 32}, num_warps=4),
        triton.Config({'BD': 32}, num_warps=8),
        triton.Config({'BD': 64}, num_warps=1),
        triton.Config({'BD': 64}, num_warps=2),
        triton.Config({'BD': 64}, num_warps=4),
        triton.Config({'BD': 64}, num_warps=8),
        triton.Config({'BD': 128}, num_warps=1),
        triton.Config({'BD': 128}, num_warps=2),
        triton.Config({'BD': 128}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def chunk_retention_fwd_kernel_h(
    k,
    v,
    h,
    initial_state,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_k = k + i_bh * T * D + i_t * BT * D + o_d
    p_v = v + i_bh * T * D + i_t * BT * D + o_d
    p_h = h + i_bh * T * D + i_t * BT * D + o_d

    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if i_t == 0:
            b_h += tl.load(initial_state + i_bh * D + o_d, mask=mask, other=0).to(tl.float32)

    for i in range(0, BT):
        mask_t = mask & ((i_t * BT + i) < T)
        b_k = tl.load(p_k, mask=mask_t, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_t, other=0).to(tl.float32)

        # Compute decay factors
        d_b = tl.exp(-0.1 * i)  # Example decay function
        d_i = tl.exp(-0.1 * (i + 1))  # Example decay function

        # Update b_h
        b_h = d_b * b_h + tl.dot(b_k, b_v)
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_t)

        p_k += D
        p_v += D
        p_h += D

    if STORE_FINAL_STATE:
        if i_t == tl.cdiv(T, BT) - 1:
            tl.store(initial_state + i_bh * D + o_d, b_h.to(initial_state.dtype.element_ty), mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BD': 32}, num_warps=1),
        triton.Config({'BD': 32}, num_warps=2),
        triton.Config({'BD': 32}, num_warps=4),
        triton.Config({'BD': 32}, num_warps=8),
        triton.Config({'BD': 64}, num_warps=1),
        triton.Config({'BD': 64}, num_warps=2),
        triton.Config({'BD': 64}, num_warps=4),
        triton.Config({'BD': 64}, num_warps=8),
        triton.Config({'BD': 128}, num_warps=1),
        triton.Config({'BD': 128}, num_warps=2),
        triton.Config({'BD': 128}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def chunk_retention_fwd_kernel_o(
    q,
    k,
    v,
    h,
    o,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_q = q + i_bh * T * D + i_t * BT * D + o_d
    p_k = k + i_bh * T * D + i_t * BT * D + o_d
    p_v = v + i_bh * T * D + i_t * BT * D + o_d
    p_h = h + i_bh * T * D + i_t * BT * D + o_d
    p_o = o + i_bh * T * D + i_t * BT * D + o_d

    b_o = tl.zeros([BD], dtype=tl.float32)
    b_s = tl.zeros([BD], dtype=tl.float32)

    for i in range(0, BT):
        mask_t = mask & ((i_t * BT + i) < T)
        b_q = tl.load(p_q, mask=mask_t, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_t, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_t, other=0).to(tl.float32)
        b_h = tl.load(p_h, mask=mask_t, other=0).to(tl.float32)

        # Compute decay factors
        d_i = tl.exp(-0.1 * (i + 1))  # Example decay function

        # Modulate contributions
        b_o += d_i * tl.dot(b_q, b_h)
        b_s += d_i * tl.dot(b_q, b_v)

        p_q += D
        p_k += D
        p_v += D
        p_h += D
        p_o += D

    b_o = b_o + b_s
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask)
