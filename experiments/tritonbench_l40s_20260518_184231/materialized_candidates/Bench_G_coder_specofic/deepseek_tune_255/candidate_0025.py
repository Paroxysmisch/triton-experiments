import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    Lk, M, N,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_z_conditional_block, stride_h_conditional_block,
    Z, H,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_Q_H: tl.constexpr, BLOCK_Q_M: tl.constexpr,
    BLOCK_K_H: tl.constexpr, BLOCK_K_N: tl.constexpr,
    BLOCK_V_H: tl.constexpr, BLOCK_V_N: tl.constexpr,
    BLOCK_O_H: tl.constexpr, BLOCK_O_N: tl.constexpr,
    ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_n = tl.program_id(2)

    Q += off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    K += off_h.to(tl.int64) * stride_kh
    V += off_h.to(tl.int64) * stride_vh
    Out += off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh + off_n.to(tl.int64) * stride_on

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_q = offs_m[:, None] * BLOCK_Q_M + (offs_n[None, :] + off_n * BLOCK_O_N) * BLOCK_Q_H
    offs_k = offs_n[None, :] * BLOCK_K_N + off_n * BLOCK_K_H
    offs_v = offs_n[None, :] * BLOCK_V_N + off_n * BLOCK_V_H

    q = tl.load(Q + offs_q * stride_qk, mask=(offs_q < M * BLOCK_Q_M), other=0.0)
    k = tl.load(K + offs_k * stride_kk, mask=(offs_k < N * BLOCK_K_N), other=0.0)
    v = tl.load(V + offs_v * stride_vn, mask=(offs_v < N * BLOCK_V_N), other=0.0)

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    if IS_CAUSAL:
        m = tl.minimum(
            tl.maximum(
                (start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) - (offs_n[None, :] + off_n * BLOCK_O_N),
                0
            ) // BLOCK_Q_M,
            Lk - 1
        )
        qk = tl.where(m[:, None] == offs_n[None, :], qk, float("-inf"))
    qk *= sm_scale
    qk = tl.where(offs_n[None, :] + off_n * BLOCK_O_N < M, qk, float("-inf"))

    lo, hi = tl.minmax(qk, axis=1)
    qk -= lo[:, None]
    qk = tl.where(offs_n[None, :] + off_n * BLOCK_O_N < M, qk, float("-inf"))

    hi = hi - lo
    denom = hi.to(tl.float32)
    qk = tl.exp2(qk - hi[:, None])

    sum_qk = tl.sum(qk, axis=1)
    o = tl.dot(qk, v)
    o = o.to(Q.dtype.element_ty)

    o = o * (1.0 / (denom + 1e-6))
    lo = lo.to(Q.dtype.element_ty)
    o = o + lo[:, None]

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_n = tl.program_id(2)

    Q += off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    K += off_h.to(tl.int64) * stride_kh
    V += off_h.to(tl.int64) * stride_vh
    Out += off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh + off_n.to(tl.int64) * stride_on

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_q = offs_m[:, None] * BLOCK_Q_M + (offs_n[None, :] + off_n * BLOCK_O_N) * BLOCK_Q_H
    offs_k = offs_n[None, :] * BLOCK_K_N + off_n * BLOCK_K_H
    offs_v = offs_n[None, :] * BLOCK_V_N + off_n * BLOCK_V_H

    q = tl.load(Q + offs_q * stride_qk, mask=(offs_q < M * BLOCK_Q_M), other=0.0)
    k = tl.load(K + offs_k * stride_kk, mask=(offs_k < N * BLOCK_K_N), other=0.0)
    v = tl.load(V + offs_v * stride_vn, mask=(offs_v < N * BLOCK_V_N), other=0.0)

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    if IS_CAUSAL:
        m = tl.minimum(
            tl.maximum(
                (start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) - (offs_n[None, :] + off_n * BLOCK_O_N),
                0
            ) // BLOCK_Q_M,
            Lk - 1
        )
        qk = tl.where(m[:, None] == offs_n[None, :], qk, float("-inf"))
    qk *= sm_scale
    qk = tl.where(offs_n[None, :] + off_n * BLOCK_O_N < M, qk, float("-inf"))

    lo, hi = tl.minmax(qk, axis=1
