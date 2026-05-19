import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    sm_scale,
    B, H, M, N, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    USE_FP8: tl.constexpr, IS_CAUSAL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    bid = pid_bh // H
    hid = pid_bh % H
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    q_ptrs = Q + bid * stride_qb + hid * stride_qh + offs_m[:, None] * stride_qm

    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    for n_offs in range(0, N, BLOCK_N):
        offs_n = n_offs + tl.arange(0, BLOCK_N)
        k_ptrs = K + bid * stride_kb + hid * stride_kh + offs_n[None, :] * stride_kn
        v_ptrs = V + bid * stride_vb + hid * stride_vh + offs_n[None, :] * stride_vn
        q_vals = tl.load(q_ptrs + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qd, mask=offs_m[:, None] < M, other=0.0)
        if USE_FP8:
            k_vals = tl.load(k_ptrs + tl.arange(0, BLOCK_DMODEL)[:, None] * stride_kd, mask=offs_n[None, :] < N, other=0).to(tl.float32)
        else:
            k_vals = tl.load(k_ptrs + tl.arange(0, BLOCK_DMODEL)
