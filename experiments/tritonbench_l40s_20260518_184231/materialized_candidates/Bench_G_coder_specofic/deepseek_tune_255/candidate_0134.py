import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, B_Start_Loc, B_Seqlen,
    Out,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_bss, stride_bse,
    n_heads: tl.constexpr,
    Lq: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // 2

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # initialize pointer to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # load Q
    q = tl.load(q_ptrs, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    # loop over k, v and update accumulator
    for start_n in range(0, block_start_loc, BLOCK_N):
        k = tl.load(k_ptrs, mask=(offs_n[None, :] < cur_batch_seq_len) & (offs_n[:, None] >= start_n), other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < cur_batch_seq_len, other=0.0)
        # compute qk
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where((offs_m[:, None] >= start_n) & (offs_n[None, :] < cur_batch_seq_len), qk, float("-inf"))
        # compute scaling factor
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # update output accumulator
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = v.to(p.dtype)
        acc += tl.dot(p.to(v.dtype), v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kbs
        v_ptrs += BLOCK_N * stride_vbs
    # initialize pointers to output
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


class ContextAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, B_Start_Loc, B_Seqlen):
        batch_size, n_heads, seq_len, d_head = q.shape
        assert q.shape == k.shape and q.shape == v.shape
        scale = d_head ** -0.5
        q = q * scale
        out = torch.empty_like(q)
        BLOCK_M = 16
        BLOCK_N = 32
        grid = (batch_size, n_heads, triton.cdiv(seq_len, BLOCK_M))
        sm_scale = 1 / (d_head ** 0.5)
        div = 1
        if k.stride(1) > 1024:
            div = 4
        elif k.stride(1) > 2048:
            div = 8
        BLOCK_DMODEL = d_head // div
        assert BLOCK_DMODEL >= 16
        kwargs = [q, k, v, sm_scale, B_Start_Loc, B_Seqlen, out]
        kwargs_str = [str(x.shape) for x in kwargs]
        kwargs_id = str(kwargs_str) + str(BLOCK_DMODEL) + str(BLOCK_M) + str(BLOCK_N)
        if kwargs_id not in context_attention_fwd._cache:
            tmp_grid = (trice(batch_size, n_heads, seq_len // BLOCK_M),)
            context_attention_fwd._cache[kwargs_id] = tmp_grid
        grid = context_attention_fwd._cache[kwargs_id]
        context_attention_fwd[grid](
            *kwargs,
            stride_qbs=q.stride(0),
            stride_qh=q.stride(1),
            stride_qd=q.stride(2),
            stride_kbs=k.stride(0),
            stride_kh=k.stride(1),
            stride_kd=k.stride(2),
            stride_vbs=v.stride(0),
            stride_vh=v.stride(1),
            stride_
