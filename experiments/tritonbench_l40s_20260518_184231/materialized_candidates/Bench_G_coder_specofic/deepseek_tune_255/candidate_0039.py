import torch
import triton
import triton.language as tl
from flash_attn.ops.triton.kv_cache_utils import *

@triton.jit
def _fwd_kernel_int8kv(
    Q, K, V, sm_scale, 
    Out, 
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    MUL_V: tl.constexpr,
    ADD_V: tl.constexpr,
    STORE_FINAL: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    P4: tl.constexpr,
    ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = tl.program_id(2)
    off_n = tl.program_id(3)

    log2e: tl.constexpr = 1.4426950408889634

    qkv_g_idx = tl.arange(0, BLOCK_M)
    m_i = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_i = off_n * BLOCK_N + tl.arange(0, BLOCK_N)

    Q_block_ptr = tl.make_block_ptr(base=Q, shape=(Z, H, N_CTX, BLOCK_K), strides=(stride_qz, stride_qh, stride_qm, stride_qk), offsets=(off_b, off_hz, m_i, qkv_g_idx), block_shape=(1, 1, BLOCK_M, BLOCK_K), order=(3, 2, 1, 0))
    K_block_ptr = tl.make_block_ptr(base=K, shape=(Z, H, BLOCK_K, BLOCK_N), strides=(stride_kz, stride_kh, stride_kn, stride_kk), offsets=(off_b, off_hz, qkv_g_idx, n_i), block_shape=(1, 1, BLOCK_K, BLOCK_N), order=(3, 2, 1, 0))
    V_block_ptr = tl.make_block_ptr(base=V, shape=(Z, H, BLOCK_K, BLOCK_N), strides=(stride_vz, stride_vh, stride_vk, stride_vn), offsets=(off_b, off_hz, qkv_g_idx, n_i), block_shape=(1, 1, BLOCK_K, BLOCK_N), order=(3, 2, 1, 0))
    m_i = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_i = off_n * BLOCK_N + tl.arange(0, BLOCK_N)
    O_block_ptr = tl.make_block_ptr(base=Out, shape=(Z, H, N_CTX, BLOCK_N), strides=(stride_oz, stride_oh, stride_om, stride_on), offsets=(off_b, off_hz, m_i, n_i), block_shape=(1, 1, BLOCK_M, BLOCK_N), order=(3, 2, 1, 0))

    qk_scale = sm_scale

    m_i_for_l = tl.arange(0, BLOCK_M)
    n_i_for_l = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if IS_CAUSAL:
        n_mask = n_i_for_l[None, :] >= m_i_for_l[:, None]

    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.int8)

    for i in range(0, tl.cdiv(K.shape[2], BLOCK_K)):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        if P2:
            q = q.to(tl.float32)
            q = (q * (1.0 / 127)).to(tl.float16)
            k = k.to(tl.float32)
            k = (k * (1.0 / 127)).to(tl.float16)
        t = tl.dot(q, k, allow_tf32=False)
        if P3:
            if P4:
                cb_mask = ((m_i_for_l[:, None] >= i * BLOCK_K) & (n_i_for_l[None, :] < (i+1) * BLOCK_K))
                t = tl.where(cb_mask, t, -10000000.0)
            else:
                t = t + (i * BLOCK_K * log2e)
        else:
            if IS_CAUSAL:
                t = t + (i * BLOCK_K * log2e)
                t = tl.where(n_mask, t, -10000000.0)
        if P2:
            t = t.to(tl.float32)
            v = v.to(tl.float32)
            v = (v * (1.0 / 127)).to(tl.float16)
            acc += tl.dot(t, v, allow_tf32=False)
        else:
            acc += tl.dot(q.to(tl.float32), (t * v).to(tl.float32), allow_tf32=False)
        K_block_ptr = tl.advance(K_block_ptr, [0, 0, BLOCK_K, 0])
        V_block_ptr = tl.advance(V_block_ptr, [0, 0, BLOCK_K, 0])
    
    if MUL_V:
        acc = acc * ADD_V

    if P2:
        acc = acc.to(tl.float16)

    if STORE_FINAL:
        tl.store(O_block_ptr, acc.to(tl.float16))
    else:
        tl.store(O_block_ptr, (acc).to(tl.float16))

def context_attention_fwd_ppl_int8kv(q, k, v, o, sm_scale, 
                                      kv_cache_c, kv_cache_k, kv_cache_v, 
                                      kv_cache_final_k, kv_cache_final_v, 
                                      kv_cache_final_o, 
                                      kv_cache_final_o_index, 
                                      kv_cache_final_o_start_m, 
                                      kv_cache_final_o_start_n, 
                                      kv_cache_final_o_n_ctx, 
                                      kv_cache_final_o_to_kv_cache_o_stride, 
                                      kv_cache_final_o_to_kv_cache_o_shape, 
                                      kv_cache_o_to_final_o
