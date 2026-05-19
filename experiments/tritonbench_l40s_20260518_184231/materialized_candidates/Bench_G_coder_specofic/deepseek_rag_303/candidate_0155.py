import torch
import triton
import triton.language as tl


@triton.jit
def chunk_linear_attn_fwd_kernel_h(
    k,  # [B, L, H, K]
    v,  # [B, L, H, V]
    h0,  # [B, H, K, V]
    h,  # [B, H, L, K, V]
    BTL: tl.constexpr,  # batch thread size along the time dim
    BTS: tl.constexpr,  # batch thread size along the softmax dim
    BK: tl.constexpr,  # batch size along the K dim
    BV: tl.constexpr,  # batch size along the V dim
    L: tl.constexpr,  # L=NK in chunk_linear_attn func
    H: tl.constexpr,  # number of heads
    K: tl.constexpr,  # K is the number of features
    V: tl.constexpr,  # V is the number of features
    USE_INITIAL_STATE: tl.constexpr,  # whether the initial state is used
):
    # thread program id
    b_B, b_H, b_L  = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_K = tl.cdiv(K, BK) 
    b_V = tl.cdiv(V, BV)
    # ignore because 0 is masked anyway
    p_h0 = tl.make_block_ptr(h0 + b_B * H * K * V, (K, V), (V, 1), (0, b_V * BV), (BK, BV), (1, 0))
    p_h = tl.make_block_ptr(h + b_L * b_B * H * K * V, (K, V), (V, 1), (b_K * BK, b_V * BV), (BK, BV), (1, 0), (b_L, b_B * H * K * V))
    o_k = tl.arange(0, BTS)
    o_v = tl.arange(0, BTS)
    for i in range(0, BTL, BTS):
        p_k = tl.make_block_ptr(k + b_B * L * H * K, (L, H, K), (H * K, K), (i, b_H, 0), (BTS, BK), (0, 1))  # [BTS, BK]
        p_v = tl.make_block_ptr(v + b_B * L * H * V, (L, H, V), (H * V, V), (0, b_H, 0), (BTS, BV), (1, 0))  # [BTS, BV]
        p_h.offsets = p_h.offsets + i * p_h.strides[0]
        if USE_INITIAL_STATE:
            # load initial state, maybe to be removed in future
            b_h0 = tl.load(p_h0, boundary_check=(0, 1))  # [BK, BV]
        b_h = tl.zeros([BTS, BV], dtype=tl.float32)
        for j in range(0, i + BTL, BTS):
            # It differs from the Chunk Linear Model due to the absence of matrix b
            # Num threads are not sufficient, therefore BTS needs to be <= BTL
            # broadcast b_h0
            if USE_INITIAL_STATE:
                b_h += tl.dot(b_h0.to(p_h0.dtype.element_ty), b_h0, allow_tf32=False)
            b_k = tl.load(p_k, boundary_check=(0, 1))  # [BTS, BK]
            b_v = tl.load(p_v, boundary_check=(0, 1))  # [BTS, BV]
            # m_s = o_k[:,None] >= o_v[None,:]
            # d_s = tl.where(m_s, 1, 0)
            b_h += tl.dot(b_k, b_v, allow_tf32=False)
            p_k = tl.advance(p_k, (0, BK))
            p_v = tl.advance(p_v, (0, BV))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        p_h = tl.advance(p_h, (BTS, 0))


@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    q,  # [B, H, L, K]
    k,  # [B, H, L, K]
    v,  # [B, H, L, V]
    h,  # [B, H, L, K, V]
    o,  # [B, H, L, V]
    s_h,  # stride size for h
    s_qk,  # stride size for q and k
    s_vo,  # stride size for v and o
    scale,  # for learned K
    BTL: tl.constexpr,  # batch thread size along the time dim
    BTS: tl.constexpr,  # batch thread size along the K dim
    BK: tl.constexpr,  # batch size along the K dim
    BV: tl.constexpr,  # batch size along the V dim
    L: tl.constexpr,  # L is the number of chunks
    H: tl.constexpr,  # number of heads
    K: tl.constexpr,  # K is the number of features
    V: tl.constexpr,  # V is the number of features
    INF: tl.constexpr,  # inf is used in masking
):
    # thread program id
    b_kv, b_bh, b_l = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_v = b_kv // (K // BK)
    b_k = b_kv % (K // BK)
    b_h = b_bh % H
    b_B = b_bh // H
    p_q = tl.make_block_ptr(q + b_B * H * L * K, (L, H, K), (H * K, K), (b_l, b_h, 0), (1, BK), (0, 1), (b_B, H * L * K))
    # p_k = tl.make_block_ptr(k + b_B * H * L * K, (H, L, K), (K, 1), (b_h, b_l, b_k * BK), (BK, 1), (0, 0))
    p_v = tl.make_block_ptr(v + b_B * H * L * V, (L, H, V), (H * V, V), (b_l, b_h, 0), (BV, 1), (0, 0))
    p_h = tl.make_block_ptr(h + b_B * H * L * K * V, (L, H, K, V), (H * K * V, V), (b_l, b_h, b_k * BK, b_v*BV), (BK * V, V), (BK, 0), (b_B*L*K*V, H*L*K*V))
    # [1, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    # b_k = tl.load(p_k, boundary_check=(0, 1)).to(b_q.dtype)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_h = tl.
