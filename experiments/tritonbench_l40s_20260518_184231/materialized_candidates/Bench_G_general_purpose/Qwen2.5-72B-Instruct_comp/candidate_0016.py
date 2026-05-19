import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out, B_Start_Loc, B_Seqlen,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_km,
    stride_vb, stride_vh, stride_vm,
    stride_ob, stride_oh, stride_om,
    sm_scale, window_size: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(B_Seqlen, BLOCK_M)
    num_pid_n = tl.cdiv(B_Seqlen, BLOCK_N)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid = pid % num_pid_in_batch
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_b = B_Start_Loc + batch_id
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + (offs_b * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qh)
    k_ptrs = K + (offs_b * stride_kb + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kh)
    v_ptrs = V + (offs_b * stride_vb + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vh)
    o_ptrs = Out + (offs_b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_oh)

    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)

    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, B_Seqlen, BLOCK_N):
        start_n = tl.max(start_n, tl.max(0, offs_m - window_size))
        start_n = tl.min(start_n, B_Seqlen - BLOCK_N)

        k = tl.load(k_ptrs + start_n * stride_km)
        qk = tl.dot(q, k, trans_b=True) * sm_scale

        m = tl.max(qk, 1)
        qk = qk - m[:, None]
        p = tl.exp(qk)
        lse_i += tl.sum(p, 1)
        p = p / lse_i[:, None]

        v = tl.load(v_ptrs + start_n * stride_vm)
        o += tl.dot(p, v)

    tl.store(o_ptrs, o)

import torch

def context_attention_fwd(Q, K, V, B_Start_Loc, B_Seqlen, sm_scale, window_size, BLOCK_M=128, BLOCK_DMODEL=64, BLOCK_N=128):
    assert Q.shape == K.shape == V.shape, "Q, K, and V must have the same shape"
    assert Q.dtype == torch.float32, "Q, K, and V must be of type float32"
    assert B_Start_Loc.dtype == torch.int32, "B_Start_Loc must be of type int32"
    assert B_Seqlen.dtype == torch.int32, "B_Seqlen must be of type int32"

    B, H, M, D = Q.shape
    Out = torch.empty_like(Q)

    grid = (B * H * (M // BLOCK_M) * (M // BLOCK_N),)

    _fwd_kernel[grid](
        Q, K, V, Out, B_Start_Loc, B_Seqlen,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        sm_scale, window_size,
        BLOCK_M, BLOCK_DMODEL, BLOCK_N
    )

    return Out
