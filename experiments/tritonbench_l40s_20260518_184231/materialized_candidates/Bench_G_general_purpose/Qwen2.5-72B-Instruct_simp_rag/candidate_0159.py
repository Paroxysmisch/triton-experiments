import torch
import triton
import triton.language as tl

@triton.jit
def chunk_linear_attn_fwd_kernel_h(
    q,  # query [B, H, T, K]
    k,  # key [B, H, T, K]
    h,  # intermediate tensor [B, H, T, T]
    s_q,  # stride size: T * K
    s_k,  # stride size: T * K
    s_h,  # stride size: T * T
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # head count
    T: tl.constexpr,  # sequence length
    K: tl.constexpr,  # feature dimension
    BTL: tl.constexpr,  # BLOCK SIZE along the sequence dimension for Q
    BTS: tl.constexpr,  # BLOCK SIZE along the sequence dimension for K
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
):
    i_h, i_b, i_c = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = tl.make_block_ptr(q + i_b * s_q, (T, K), (s_q, 1), (i_c * BTL, 0), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_b * s_k, (T, K), (s_k, 1), (0, 0), (BTS, BK), (1, 0))
    p_h = tl.make_block_ptr(h + i_b * s_h, (T, T), (s_h, 1), (i_c * BTL, 0), (BTL, BTS), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_h = tl.zeros([BTL, BTS], dtype=tl.float32)

    for _ in range(0, T, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h += tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))

    tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    q,  # query [B, H, T, K]
    k,  # key [B, H, T, K]
    v,  # value [B, H, T, V]
    h,  # intermediate tensor [B, H, T, T]
    o,  # output [B, H, T, V]
    s_q,  # stride size: T * K
    s_k,  # stride size: T * K
    s_v,  # stride size: T * V
    s_h,  # stride size: T * T
    s_o,  # stride size: T * V
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # head count
    T: tl.constexpr,  # sequence length
    K: tl.constexpr,  # feature dimension
    V: tl.constexpr,  # feature dimension
    BTL: tl.constexpr,  # BLOCK SIZE along the sequence dimension for Q
    BTS: tl.constexpr,  # BLOCK SIZE along the sequence dimension for K
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
):
    i_h, i_b, i_c = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = tl.make_block_ptr(q + i_b * s_q, (T, K), (s_q, 1), (i_c * BTL, 0), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_b * s_k, (T, K), (s_k, 1), (0, 0), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_b * s_v, (T, V), (s_v, 1), (0, 0), (BTS, BV), (1, 0))
    p_h = tl.make_block_ptr(h + i_b * s_h, (T, T), (s_h, 1), (i_c * BTL, 0), (BTL, BTS), (1, 0))
    p_o = tl.make_block_ptr(o + i_b * s_o, (T, V), (s_o, 1), (i_c * BTL, 0), (BTL, BV), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_h = tl.load(p_h, boundary_check=(0, 1))
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)

    for _ in range(0, T, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_o += tl.dot(b_s, b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (BTS, 0))

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
