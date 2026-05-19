import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, B_Start_Loc, B_Seqlen,
    Out,
    stride_bq, stride_sq, stride_hq, stride_dq,
    stride_bk, stride_sk, stride_hk, stride_dk,
    stride_bv, stride_sv, stride_hv, stride_dv,
    stride_bo, stride_so, stride_ho, stride_do,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_batch_in_all_start_at = tl.load(B_Start_Loc + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch_in_all_start_at + offs_m[:, None]) * stride_bq + cur_head * stride_hq + offs_d[None, :] * stride_dq
    off_k = offs_n[None, :] * stride_bk + cur_head * stride_hk + offs_d[:, None] * stride_dk
    off_v = offs_n[:, None] * stride_bv + cur_head * stride_hv + offs_d[None, :] * stride_dv

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + (cur_batch_in_all_start_at + start_n) * stride_sk, mask=(start_n + offs_n[None, :]) < cur_batch_seq_len, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

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

        v = tl.load(v_ptrs + (cur_batch_in_all_start_at + start_n) * stride_sv, mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)

        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new

    off_o = (cur_batch_in_all_start_at + offs_m[:, None]) * stride_bo + cur_head * stride_ho + offs_d[None, :] * stride_do
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    return

@torch.no_grad()
def context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len):
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]

    BLOCK_M = 128
    BLOCK_N = 64 if Lk <= 64 else 32

    # 128, 128, 64 -> 128, 64, 64
    if Lk > 64:
        q = q.reshape(batch, head, BLOCK_M, BLOCK_M).transpose(2, 3)
        v = v.reshape(batch, head, BLOCK_M, BLOCK_M).transpose(2, 3)
        k = k.reshape(batch, head, BLOCK_M, BLOCK_M)

    else:
        q = q.reshape(batch, head, BLOCK_M, BLOCK_M)
        k = k.reshape(batch, head, BLOCK_M, BLOCK_M)
        v = v.reshape(batch, head, BLOCK_M, BLOCK_M).transpose(2, 3)

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK_M))

    num_warps = 4 if Lk <= 64 else 8

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        q.shape[0],
        q.shape[1],
        max_input_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=Lk,
        num_warps=num_warps,
        num_stages=1,
    )
    return
