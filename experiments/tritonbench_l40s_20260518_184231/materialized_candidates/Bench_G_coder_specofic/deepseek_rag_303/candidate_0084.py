import triton
import triton.language as tl

from triton import compile, wrap
from triton.language.extra.cuda.libdevice import exp2 as libdevice_exp2


try:
    from triton.language.math import exp2 as _exp2
except ImportError:
    _exp2 = libdevice_exp2


@triton.jit
def _fwd_kernel_aligned(
    Q,
    K,
    V,
    B0,
    Out,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_b0n,
    stride_b0h,
    kv_group_num: tl.constexpr,
    BIAS_LAST_SIZE: tl.constexpr,
    USE_BIAS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]
    off_b0 = (
        offs_n[None, :] * stride_b0n
        + cur_kv_head * stride_b0h
        + offs_d[:, None]
    )

    q = tl.load(
        Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0
    )  # NOTE: not needed, might violate L2, if set size too large, need to ensure safe here
    q = tl.trans(q)  # make it (BLOCK_DMODEL, BLOCK_M) so that axis name will match with k
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    b0_ptrs = (
        B0 + off_b0
    )  # TODO: what if we uniformly add for all kv_heads instead of head-wise (may be suboptimal)

    # cache current k and v, which are necessary, or we can make it rolling
    k = tl.load(
        k_ptrs,
        mask=(start_m * BLOCK_M + offs_n[None, :]) < cur_batch_seq_len,
        other=0.0,
    )
    v = tl.load(
        v_ptrs,
        mask=(start_m * BLOCK_M + offs_n[:, None]) < cur_batch_seq_len,
        other=0.0,
    )

    if USE_BIAS:
        b0 = tl.load(b0_ptrs, mask=start_m * BLOCK_M + offs_n[None, :] < BIAS_LAST_SIZE)
    else:
        b0 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # scale k and add bias
        k_shift = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
            other=0.0,
        )
        k_shift = (k_shift * sm_scale + b0) if USE_BIAS else (k_shift * sm_scale)

        # k, v are now in column major
        # Step 1: Compute qk
        qk = tl.dot(q, k_shift)
        qk = tl.where(offs_m[:, None] >= start_n, qk, float("-inf"))

        # Step 2: compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)  # always valid since we initialized with -inf
        p = _exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = _exp2(m_i - m_i_new)
        beta = _exp2(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
            other=0.0,
        )
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    acc = tl.trans(acc)  # trans back to (BLOCK_M, BLOCK_DMODEL)
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0
    )  # NOTE: other=0.0 not necessarily
