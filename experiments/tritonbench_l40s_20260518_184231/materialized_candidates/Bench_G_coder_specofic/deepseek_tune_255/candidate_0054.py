import torch
import triton
import triton.language as tl

@triton.jit
def chunk_gated_abc_fwd_kernel_cum(
    s, o,
    stride_s_b, stride_s_h, stride_s_t, stride_s_d,
    stride_o_b, stride_o_h, stride_o_t, stride_o_d,
    T: tl.constexpr, S: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr,
):
    # Kernel function to compute cumulative sum
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1) % BT, tl.program_id(1) // BT

    p_s = tl.make_block_ptr(
        base=s,
        shape=(T, S, -1),
        strides=(stride_s_t, stride_s_d, stride_s_b),
        offsets=(i_t * BT, 0, i_bh * BS),
        block_shape=(BT, -1),
        order=(1, 0),
    )
    p_o = tl.make_block_ptr(
        base=o,
        shape=(T, S, -1),
        strides=(stride_o_t, stride_o_d, stride_o_b),
        offsets=(i_t * BT, 0, i_bh * BS),
        block_shape=(BT, -1),
        order=(1, 0),
    )
    m_s = i_t * BT + tl.arange(0, BT) < T

    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k, v, g, h0, ht, h,
    stride_k_b, stride_k_h, stride_k_t, stride_k_d,
    stride_v_b, stride_v_h, stride_v_t, stride_v_d,
    stride_g_b, stride_g_h, stride_g_t, stride_g_d,
    stride_ht_b, stride_ht_h, stride_ht_t, stride_ht_d,
    stride_h_b, stride_h_h, stride_h_t, stride_h_d,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
):
    # Kernel function to compute gated cumulative sum
    i_bh, i_t = tl.program_id(1), tl.program_id(0)

    p_k = tl.make_block_ptr(
        base=k,
        shape=(T, K, -1),
        strides=(stride_k_t, stride_k_d, stride_k_b),
        offsets=(i_t * BT + tl.arange(0, BT), 0, i_bh * BK),
        block_shape=(BT, BK),
        order=(1, 0),
    )
    p_v = tl.make_block_ptr(
        base=v,
        shape=(T, V, -1),
        strides=(stride_v_t, stride_v_d, stride_v_b),
        offsets=(i_t * BT + tl.arange(0, BT), 0, i_bh * BV),
        block_shape=(BT, BV),
        order=(1, 0),
    )
    p_g = tl.make_block_ptr(
        base=g,
        shape=(T, K, -1),
        strides=(stride_g_t, stride_g_d, stride_g_b),
        offsets=(i_t * BT + tl.arange(0, BT), 0, i_bh * BK),
        block_shape=(BT, BK),
        order=(1, 0),
    )
    p_h = tl.make_block_ptr(
        base=h,
        shape=(T, K, -1),
        strides=(stride_h_t, stride_h_d, stride_h_b),
        offsets=(i_t * BT + tl.arange(0, BT), 0, i_bh * BK),
        block_shape=(BT, BK),
        order=(1, 0),
    )
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            base=h0,
            shape=(K, -1),
            strides=(stride_ht_d, stride_ht_b),
            offsets=(0, i_bh * BK),
            block_shape=(BK,),
            order=(0,),
        )
        b_h0 = tl.load(p_h0, boundary_check=(0,)).to(tl.float32)
    else:
        b_h0 = tl.zeros((BT, BK), dtype=tl.float32)
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            base=ht,
            shape=(K, -1),
            strides=(stride_ht_d, stride_ht_b),
            offsets=(0, i_bh * BK),
            block_shape=(BK,),
            order=(0,),
        )
    b_h = tl.zeros((BT, BK), dtype=tl.float32)
    b_h += b_h0
    for i in range(0, tl.cdiv(T, BT)):
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_h = b_h * b_g + b_k * b_v
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        p_k = tl.advance(p_k, (BT, 0))
        p_v = tl.advance(p_v, (BT, 0))
        p_g = tl.advance(p_g, (BT, 0))
        p_h = tl.advance(p_h, (BT, 0))
        if USE_INITIAL_STATE:
            p_h0 = tl.advance(p_h0, (BK,))
        if STORE_FINAL_STATE:
            p_ht = tl.advance(p_ht, (BK,))
        if USE_INITIAL_STATE:
            b_h0 = tl.load(p_h0, boundary_check=(0,)).to(tl.float32)
        if STORE_FINAL_STATE:
            tl.store(p_ht, b_h.to(p_ht.dtype
