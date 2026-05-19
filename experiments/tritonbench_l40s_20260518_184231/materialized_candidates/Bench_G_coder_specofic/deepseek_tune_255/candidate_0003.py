import torch
import triton
import triton.language as tl
from packaging import version

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,  #
    B_Loc, B_Seqlen,  #
    B_Start_Loc,  #
    Out,  #
    max_batch_seq_len, max_head_size,  #
    stride_b_loc_b, stride_b_loc_s,  #
    stride_qbs, stride_qh, stride_qd,  #
    stride_kbs, stride_kh, stride_kd,  #
    stride_vbs, stride_vh, stride_vd,  #
    stride_obs, stride_oh, stride_od,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,  #
    BLOCK_DMODEL_PADDED: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    cur_batch_qk_elements_num = BLOCK_DMODEL_PADDED * cur_batch_seq_len
    cur_batch_qk_num = cur_batch_seq_len // BLOCK_M
    cur_batch_qk_elements_num = cur_batch_qk_num * BLOCK_DMODEL_PADDED

    offs_n = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_b = cur_batch_in_all_start_index + tl.arange(0, BLOCK_N)
    offs_q = (cur_head * stride_qh + offs_b[:, None] * stride_qbs + (offs_d[None, :] * stride_qd))

    q = tl.load(Q + offs_q)

    k_ptrs = K + cur_head * stride_kh
    v_ptrs = V + cur_head * stride_vh

    lo = 0
    hi = cur_batch_qk_num
    if cur_batch_qk_num > 0:
        qk = tl.zeros([BLOCK_DMODEL_PADDED, BLOCK_M], dtype=tl.float32)
        for start_n in range(lo, hi, BLOCK_N):
            offs_k = (offs_b[:, None] * stride_kbs + (offs_n[None, :] * stride_kd))
            k = tl.load(k_ptrs + offs_k, mask=(offs_b[:, None] < cur_batch_seq_len) & (offs_n[None, :] < cur_batch_seq_len), other=0.0)
            qk += tl.dot(q, k)
        qk *= sm_scale

        if version.parse(torch.__version__) >= version.parse("1.9.0"):
            qk = tl.where((offs_n[:, None] >= cur_batch_seq_len), float("-inf"), qk)
        else:
            qk = tl.where((offs_n[:, None] >= cur_batch_seq_len), float("-1e20"), qk)

        lo = 0
        hi = cur_batch_qk_num

    offs_n = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_b = cur_batch_in_all_start_index + tl.arange(0, BLOCK_N)
    offs_o = (cur_head * stride_oh + offs_b[:, None] * stride_obs + (offs_d[None, :] * stride_od))

    out = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = (offs_n[:, None] * stride_kbs + (offs_d[None, :] * stride_kd))
        k = tl.load(k_ptrs + offs_k, mask=(offs_n[:, None] < cur_batch_seq_len), other=0.0)

        offs_v = (offs_n[:, None] * stride_vbs + (offs_d[None, :] * stride_vd))
        v = tl.load(v_ptrs + offs_v, mask=(offs_n[:, None] < cur_batch_seq_len), other=0.0)

        v = v.to(tl.float32)

        m = tl.dot(qk, k)
        m = tl.where((offs_n[None, :] >= cur_batch_seq_len), float("-inf"), m)
        m = tl.where((offs_n[:, None] >= cur_batch_seq_len), float("-inf"), m)

        m = tl.where((offs_n[None, :] >= cur_batch_seq_len) | (offs_n[:, None] >= cur_batch_seq_len), float("-inf"), m)
        m = tl.where((offs_b[:, None] + offs_n[None, :]) >= cur_batch_seq_len, float("-inf"), m)

        m = tl.where((offs_b[:, None] + offs_n[None, :]) < max_batch_seq_len, m, float("-inf"))

        m = tl.where((offs_b[:, None] + offs_n[None, :]) >= cur_batch_in_all_start_index, m, float("-inf"))

        m = tl.where((offs_b[:, None] + offs_n[None, :]) < (cur_batch_in_all_start_index + cur_batch_seq_len), m, float("-inf"))

        m = tl.where((offs_b[:, None] + offs_n[None, :]) >= cur_batch_in_all_start_index, m, float("-inf"))

        m = tl.where((offs_b[:, None] + offs_n[None, :]) < (cur_batch_in_all_start_index + cur_batch_seq_len), m, float("-inf"))

        lo = 0
        hi = cur_batch_qk_num
        ls = tl.cumsum(tl.log(tl.sum(tl.exp(m), axis=1)))
        m = m - ls[:, None]

        m = tl.where((offs_n[None, :] >= cur_batch_seq_len) | (offs_n[:, None] >= cur_batch_seq_len), float("-inf"), m)

        m = tl.where((offs_b[:, None] + offs_n[None, :]) >= cur_batch_in_all_start_index, m, float("-inf"))

        m = tl.where((offs_b[:, None] + offs_n[None, :]) < (cur_batch_in_all_start_index + cur_batch_seq_len), m, float("-inf"))

        alpha = tl.exp(m)
        out += tl.dot(alpha, v)

    tl.store(Out + offs_o, out)


def
