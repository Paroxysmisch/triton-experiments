import torch
import triton
import triton.language as tl

# Triton kernel for forward pass
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, B_Start_Loc, B_Seqlen, Out, 
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_batch_range, batch_range_size,
    block_n_context,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_seq_block = tl.program_id(2)
    cur_q_seq_block = cur_kv_seq_block // block_n_context

    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    block_n_limit = tl.cdiv(cur_batch_seq_len, BLOCK_N)
    block_n = cur_kv_seq_block % block_n_limit

    block_m_limit = tl.cdiv(cur_batch_seq_len, BLOCK_M)
    block_m_idx = cur_q_seq_block // block_m_limit
    block_m = cur_q_seq_block % block_m_limit

    offs_q = cur_batch * stride_qbs + cur_head * stride_qh + block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = cur_head * stride_kh + block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_v = cur_head * stride_vh + block_n * BLOCK_N + tl.arange(0, BLOCK_N)

    q = tl.load(Q + offs_q)

    k_ptrs = K + offs_k
    v_ptrs = V + offs_v

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for _ in range(0, block_m_idx + 1):
        k_window_end = tl.load(B_Start_Loc + cur_batch + 1)
        k_start_bound = (cur_batch_start_loc + block_m * BLOCK_M)
        k_end_bound = tl.minimum(k_window_end, cur_batch_start_loc + (block_m + 1) * BLOCK_M)

        k_bound_start = tl.maximum(k_start_bound, cur_batch_start_loc + block_n * BLOCK_N)
        k_bound_end = tl.minimum(k_end_bound, cur_batch_start_loc + (block_n + 1) * BLOCK_N)

        k_bound_len = k_bound_end - k_bound_start

        k_off = k_ptrs + (k_bound_start - cur_batch_start_loc) + tl.arange(0, k_bound_len)
        v_off = v_ptrs + (k_bound_start - cur_batch_start_loc) + tl.arange(0, k_bound_len)

        k = tl.load(k_off)
        v = tl.load(v_off)

        qk = tl.zeros([BLOCK_M, k_bound_len], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        qk = tl.where(k_off < cur_batch_start_loc + block_n * BLOCK_N, qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]

        v = v.to(p.dtype)
        acc += tl.dot(p, v)

        l_i = l_i_new
        m_i = m_i_new

        block_m += block_m_limit

    tl.store(Out + cur_batch * stride_obs + cur_head * stride_oh + block_m * BLOCK_M + tl.arange(0, BLOCK_M), acc)
    return

# Function to call the Triton kernel
@torch.no_grad()
def context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len, block_n_context, kv_stride_h, kv_stride_d, kv_stride_bs, kv_stride_n, num_heads, softmax_scale):
    BLOCK_M = 128
    BLOCK_N = 32
    BLOCK_DMODEL = v.shape[-1]

    grid = (q.shape[0], num_heads, triton.cdiv(max_input_len, BLOCK_N) * triton.cdiv(max_input_len, BLOCK_M))

    num_warps = 4 if BLOCK_DMODEL <= 128 else 8

    _fwd_kernel[grid](
        q, k, v, softmax_scale, b_start_loc, b_seq_len, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        b_start_loc.stride(0), b_start_loc.shape[0],
        block_n_context,
        BLOCK_M, BLOCK_DMODEL, BLOCK_N,
        num_warps=num_warps,
        num_stages=1,
    )
    return
