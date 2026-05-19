import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    L,
    M,
    Y,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
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
def _bwd_prep(
    Y,
    DY,
    L,
    NewDY,
    Delta,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    y = tl.load(Y + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    dy = tl.load(DY + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    denom = tl.load(L + off_m).to(tl.float32)
    dy = dy / denom[:, None]
    delta = tl.sum(y * dy, axis=1)
    tl.store(NewDY + off_m[:, None] * D_HEAD + off_n[None, :], dy)
    tl.store(Delta + off_m, delta)

@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Y,
    DY,
    DQ,
    DK,
    DV,
    L,
    M,
    D,
    Z,
    H,
    N_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    o_zh = tl.program_id(0)
    o_z = o_zh // H
    o_h = o_zh % H
    s_qz, s_qh, s_qm, s_qk = Q.stride()
    _, _, s_kn, s_kk = K.stride()
    off = o_z * s_qz + o_h * s_qh
    offs_k = tl.arange(0, BLOCK_K)
    for i in range(0, num_block):
        i *= BLOCK_M
        offs_m = i + tl.arange(0, BLOCK_M)
        offs_n = i + tl.arange(0, BLOCK_M)
        qs = Q + off + (offs_m[:, None] * s_qm + offs_k[None, :] * s_qk)
        ks = K + off + (offs_n[:, None] * s_kn + offs_k[None, :] * s_kk)
        vs = V + off + (offs_n[:, None] * s_qm + offs_k[None, :] * s_qk)
        dqs = DQ + off + (offs_m[:, None] * s_qm + offs_k[None, :] * s_qk)
        dys = DY + off + (offs_m[:, None] * s_qm + offs_k[None, :] * s_qk)
        ds = D + o_zh * N_CTX
        ms = M + o_zh * N_CTX
        dv = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        k = tl.load(ks)
        v = tl.load(vs)
        for j in range(i, num_block * BLOCK_M, BLOCK_M):
            j += tl.arange(0, BLOCK_N)
            q = tl.load(qs)
            qk = tl.dot(q, tl.trans(k))
            qk = tl.where(j[:, None] >= (offs_n[None, :]), qk, float("-inf"))
            m = tl.load(ms + j)
            p = tl.exp(qk * sm_scale - m[:, None])
            dy = tl.load(dys)
            dv += tl.dot(
                tl.trans(p.to(Q.dtype.element_ty)), dy
            )
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - tl.load(ds + j)[:, None]
            dp += tl.dot(dy, tl.trans(v))
            ds = p * dp * sm_scale
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
            dq = tl.load(dqs)
            dq += tl.dot(ds.to(Q.dtype.element_ty), k)
            tl.store(dqs, dq)
            qs += BLOCK_M * s_qm
            dqs += BLOCK_M * s_qm
            dys += BLOCK_M * s_qm
        tl.store(DK + off + (offs_n[:, None] * s_kn + offs_k[None, :] * s_kk), dk)
        tl.store(DV + off + (offs_n[:, None] * s_qm + offs_k[None, :] * s_qk), dv)
