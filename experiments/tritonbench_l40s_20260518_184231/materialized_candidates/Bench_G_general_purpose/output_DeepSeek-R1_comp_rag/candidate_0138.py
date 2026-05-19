import torch
import triton
import triton.language as tl

@triton.jit
def chunk_linear_attn_fwd_kernel_h(
    # Tensors
    k, v, h, initial_state,
    # Strides
    s_k_h, s_k_t, s_k_d,
    s_v_h, s_v_t, s_v_d,
    s_h_h, s_h_t, s_h_d,
    # Constants
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr
):
    # Grid indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), 
                           (0, i_k * BK), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), 
                           (0, i_v * BV), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_h_h, (T, V), (s_h_t, s_h_d), 
                           (0, i_v * BV), (BT, BV), (1, 0))
    
    # Initialize h with initial state if provided
    if USE_INITIAL_STATE:
        p_init = tl.make_block_ptr(initial_state + i_bh * V, (V,), (1,), 
                                  (i_v * BV,), (BV,), (0,))
        h_init = tl.load(p_init, boundary_check=(0,))
        h_prev = tl.broadcast(h_init, [BT, BV])
    else:
        h_prev = tl.zeros([BT, BV], dtype=tl.float32)
    
    for t in range(0, tl.cdiv(T, BT)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        
        # Compute outer product and update h
        h_update = tl.dot(b_k, b_v, allow_tf32=False)
        h_prev = h_prev + h_update
        tl.store(p_h, h_prev.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        
        p_k = tl.advance(p_k, (BT, 0))
        p_v = tl.advance(p_v, (BT, 0))
        p_h = tl.advance(p_h, (BT, 0))

@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    q, k, h, o,
    s_q_h, s_q_t, s_q_d,
    s_k_h, s_k_t, s_k_d,
    s_h_h, s_h_t, s_h_d,
    s_o_h, s_o_t, s_o_d,
    scale: tl.constexpr,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    MASK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    
    p_q = tl.make_block_ptr(q + i_bh * s_q_h, (K, T), (s_q_d, s_q_t), 
                           (i_k * BK, 0), (BK, BT), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), 
                           (0, i_k * BK), (BT, BK), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_h_h, (T, V), (s_h_t, s_h_d), 
                           (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_o_h, (T, V), (s_o_t, s_o_d), 
                           (0, i_v * BV), (BT, BV), (1, 0))
    
    acc = tl.zeros([BT, BV], dtype=tl.float32)
    b_q = tl.load(p_q, boundary_check=(0, 1)) * scale
    b_k = tl.load(p_k, boundary_check=(0, 1))
    
    # Compute attention scores
    attn = tl.dot(b_q, b_k, allow_tf32=False)
    if MASK:
        mask = o_i[:, None] >= tl.arange(0, BT)[None, :]
        attn = tl.where(mask, attn, float('-inf'))
    
    attn = tl.softmax(attn, axis=1)
    b_h = tl.load(p_h, boundary_check=(0, 1))
    acc = tl.dot(attn, b_h, allow_tf32=False)
    
    tl.store(p_o, acc.to(p_o.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def chunk_linear_attn
