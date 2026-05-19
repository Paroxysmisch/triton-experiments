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


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        assert torch.cuda.get_device_capability()[0] > 7
        BLOCK = 128
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        y = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8
        tmp = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            L,
            m,
            y,
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_K=Lk,
            num_warps=num_warps,
            num_stages=2,
        )

        ctx.save_for_backward(q, k, v, y, L, m)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_M = BLOCK
        ctx.BLOCK_N = BLOCK
        ctx.BLOCK_K = Lk
        return y

    @staticmethod
    def backward(ctx, dy):
        q, k, v, y, l, m = ctx.saved_tensors
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dy = dy.contiguous()
        dy_scaled = torch.empty_like(dy)
        delta = torch.empty_like(l)
        _bwd_prep[(ctx.grid[0] * ctx.grid[1],)](
            y,
            dy,
            l,
            dy_scaled,
            delta,
            BLOCK_M=ctx.BLOCK_M,
            D_HEAD=ctx.BLOCK_K,
        )
        _bwd_kernel[(ctx.grid[1],)](
            q,
            k,
            v,
            ctx.sm_scale,
            y,
            dy_scaled,
            dq,
            dk,
            dv,
            l,
