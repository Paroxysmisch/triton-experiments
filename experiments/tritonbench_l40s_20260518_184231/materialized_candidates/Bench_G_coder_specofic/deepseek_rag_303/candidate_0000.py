import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, L, M, Y, Z, H, N_CTX, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    start = tl.program_id(0)
    off = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_K)
    offs_m = start * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    _, s_qh, s_qm, s_qk = Q.stride()
    _, _, s_kn, s_kk = K.stride()
    _, _, s_vk, _ = V.stride()
    _, s_yh, s_ym, s_yn = Y.stride()
    q = tl.load(Q + off * s_qh + offs_m[:, None] * s_qm + offs_d[None, :] * s_qk)
    ks = K + off * s_qh + offs_n[None, :] * s_kn + offs_d[:, None] * s_kk
    vs = V + off * s_qh + offs_n[:, None] * s_qm + offs_d[None, :] * s_qk
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    y = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    for i in range(0, (start + 1) * BLOCK_M, BLOCK_N):
        k = tl.load(ks + i * s_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (i + offs_n[None, :]), qk, float("-inf"))

        m2 = tl.maximum(tl.max(qk, 1), m)
        l *= tl.exp(m - m2)
        p = tl.exp(qk - m2[:, None])
        l2 = tl.sum(p, 1) + l
        l3 = 1.0 / l2
        p *= l3[:, None]
        y *= (l * l3)[:, None]
        v = tl.load(vs + i * s_vk)
        p = p.to(Q.dtype.element_ty)
        y += tl.dot(p, v)
        l = l2
        m = m2

        m2 = tl.max(qk, 1)
        p = tl.exp(qk - m2[:, None])
        m3 = tl.maximum(m, m2)
        alpha = tl.exp(m - m3)
        beta = tl.exp(m2 - m3)
        l2 = alpha * l + beta * tl.sum(p, 1)
        p_scale = beta / l2
        p = p * p_scale[:, None]
        y_scale = l / l2 * alpha
        y = y * y_scale[:, None]
        v = tl.load(vs + i * s_vk)
        p = p.to(v.dtype)
        y += tl.dot(p, v)
        l = l2
        m = m3

    tl.store(L + off * N_CTX + offs_m, l)
    tl.store(M + off * N_CTX + offs_m, m)
    tl.store(Y + off * s_yh + offs_m[:, None] * s_ym + offs_d[None, :] * s_yn, y)


@triton.jit
def _bwd_intra_kernel(Q, K, V, sm_scale, Y, DO, DQ, DK, DV, L, M, H, N_CTX, BLOCK: tl.constexpr, CBLOCK: tl.constexpr):
    off_d = tl.program_id(0)
    off_zh = tl.program_id(1)
    off_z = off_zh // H
    off_h = off_zh % H
    offs_d = off_d * CBLOCK + tl.arange(0, CBLOCK)
    offs_m = tl.arange(0, BLOCK)
    _, s_qh, s_qm, s_qk = Q.stride()
    _, _, s_kn, s_kk = K.stride()
    _, _, s_vk, _ = V.stride()
    _, s_yh, s_ym, s_yn = Y.stride()
    _, s_dh, s_dm, s_dn = DO.stride()
    dqs = DQ + off_d * s_qh + offs_m[:, None] * s_qm + offs_d[None, :] * s_qk
    ks = K + off_h * s_qh + off_z * s_kn + offs_d[:, None] * s_qm
    vs = V + off_h * s_qh + off_z * s_vk + offs_d[None, :] * s_qm
    dys = DO + off_d * s_dh + offs_m[:, None] * s_dm + offs_d[None, :] * s_dn
    dqs_lst = [dqs + (i * s_qh + offs_m[:, None] * s_qm + offs_d[None, :] * s_qk) for i in range(int(off_d))]
    q = tl.load(Q + off_h * s_qh + off_z * s_qh + offs_m[:, None] * s_qm + offs_d[None, :] * s_qk)
    dq = tl.zeros([BLOCK, CBLOCK], dtype=tl.float32)
    dk = tl.zeros([CBLOCK, CBLOCK], dtype=tl.float32)
    dv = tl.zeros([CBLOCK, CBLOCK], dtype=tl.float32)
    y = tl.load(Y + (off_z * H + off_h) * s_yh + off_zh * s_ym + offs_m[:, None] * s_ym + offs_d[None, :] * s_yn)
    dy = tl.load(dys)
    l = tl.load(L + off_z * N_CTX + offs_m)
    m = tl.load(M + off_z * N_CTX + offs_m)
    denom = tl.exp(m - l)
    for _ in range(int(off_d)):
        dq += tl.load(dqs_lst.pop())
    p = tl.exp(q * sm_scale - l[:, None])
    p = p.to(Q.dtype.element_ty)
    d = p * dy
    p = p * denom[:, None]
    dk += tl.trans(tl.dot(tl.trans(d), tl.load(ks)))
    dv += tl.dot(p, tl.trans(v))
    dq += tl.dot(tl.trans(d), y)
    tl.store(DQ + off_d * s_qh + offs_m[:, None] * s_qm + offs_d[None, :] * s_q
