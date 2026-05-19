import torch
import triton
import triton.language as tl
from packaging import version

@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
    q, k, v, w, u, o, initial_state, final_state,
    scale, T, K, V,
    stride_qkv_h, stride_qkv_t, stride_qkv_d,
    stride_qk_h, stride_qk_t,
    stride_kv_h, stride_kv_t, stride_kv_d,
    stride_kw_h, stride_kw_t,
    stride_u_h, stride_u_t,
    stride_ow_h, stride_ow_t,
    stride_oh_h, stride_oh_t,
    stride_final_state_h, stride_final_state_t,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    REVERSE: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([V], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h = tl.load(initial_state + i_bh * stride_final_state_h + i_k[None] * stride_final_state_t, mask=i_k[None] < K, other=0.0).to(tl.float32)
    b_w = tl.zeros([K, V], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, 4)):
        i_t = i * 4
        if REVERSE:
            p_q = tl.make_block_ptr(q + i_bh * stride_qkv_h + (i_k + i_t * K)[:, None] * stride_qkv_t, (4, K), (stride_qkv_h, stride_qkv_t), (i_t, 0), (4, 1), (T, K))
            p_k = tl.make_block_ptr(k + i_k[None, :] * stride_qk_h + i_t * stride_qk_t, (K, 4), (stride_qk_h, stride_qk_t), (0, i_t), (1, 4), (K, T))
            p_v = tl.make_block_ptr(v + i_bh * stride_kv_h + (i_k + i_t * K)[:, None] * stride_kv_t, (4, K), (stride_kv_h, stride_kv_t), (i_t, 0), (4, 1), (T, K))
            p_w = tl.make_block_ptr(w + i_bh * stride_kw_h + (i_k + i_t * K)[:, None] * stride_kw_t, (4, K), (stride_kw_h, stride_kw_t), (i_t, 0), (4, 1), (T, K))
            p_u = tl.make_block_ptr(u + i_k[None, :] * stride_u_h + i_t * stride_u_t, (K, 4), (stride_u_h, stride_u_t), (0, i_t), (1, 4), (K, T))
        else:
            p_q = tl.make_block_ptr(q + i_bh * stride_qkv_h + (i_k + i_t * K)[:, None] * stride_qkv_t, (4, K), (stride_qkv_h, stride_qkv_t), (i_t, 0), (4, 1), (T, K))
            p_k = tl.make_block_ptr(k + i_k[None, :] * stride_qk_h + i_t * stride_qk_t, (K, 4), (stride_qk_h, stride_qk_t), (0, i_t), (1, 4), (K, T))
            p_v = tl.make_block_ptr(v + i_bh * stride_kv_h + (i_k + i_t * K)[:, None] * stride_kv_t, (4, K), (stride_kv_h, stride_kv_t), (i_t, 0), (4, 1), (T, K))
            p_w = tl.make_block_ptr(w + i_bh * stride_kw_h + (i_k + i_t * K)[:, None] * stride_kw_t, (4, K), (stride_kw_h, stride_kw_t), (i_t, 0), (4, 1), (T, K))
            p_u = tl.make_block_ptr(u + i_k[None, :] * stride_u_h + i_t * stride_u_t, (K, 4), (stride_u_h, stride_u_t), (0, i_t), (1, 4), (K, T))
        mask_t = (i_t + tl.arange(0, 4)) < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_w = b_w * 0.0 + tl.load(p_w, boundary_check=(0, 1))
        b_u = tl.load(p_u, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_k.dtype)
        b_o = tl.dot(b_q, b_k, allow_tf32=False)
        b_o = b_o + tl.dot(b_w, b_u.to(b_w.dtype), allow_tf32=False)
        b_o = b_o.to(b_v.dtype)
        b_h = b_h * 0.0 + tl.dot(b_k, b_v, allow_tf32=False)
        if REVERSE:
            p_o = tl.make_block_ptr(o + (i_k + i_t * K)[:, None] * stride_ow_h + i_bh * stride_ow_t, (4, V), (stride_ow_h, stride_ow_t), (i_t, 0), (4, 1), (T, V))
            p_h = tl.make_block_ptr(o + (i_k + i_t * K)[:, None] * stride_oh_h + i_bh * stride_oh_t, (4, V), (stride_oh_h, stride_oh_t), (i_t, 0), (4, 1), (T, V))
        else:
            p_o = tl.make_block_ptr(o + (i_k + i_t * K)[:, None] * stride_ow_h + i_bh * stride_ow_t, (4, V), (stride_ow_h, stride_ow_t), (i_t, 0), (4, 1), (T, V))
            p_h = tl.make_block_ptr(o + (i_k + i_t * K)[:, None] * stride_oh_h + i_bh * stride_oh_t,
