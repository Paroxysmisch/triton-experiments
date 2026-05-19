import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    Q, K, V, O,
    ACC, SCALE, QSCALE, KSCALE,
    stride_qz, stride_qh, stride_qm, stride_qq,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_oo,
    stride_accz, stride_acch, stride_accm, stride_accn,
    B, H, N_CTX, D_HEAD,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr
):
    bid_z = tl.program_id(0)
    bid_h = tl.program_id(1)
    block_m = tl.program_id(2) * BLOCK_M
    offs_m = block_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    q_ptrs = Q + bid_z * stride_qz + bid_h * stride_qh + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qq
    k_ptrs = K + bid_z * stride_kz + bid_h * stride_kh + offs_m[:, None] * stride_kn + offs_n[None, :] * stride_kk
    v_ptrs = V + bid_z * stride_vz + bid_h * stride_vh + offs_n[:, None] * stride_vn + tl.arange(0, D_HEAD)[None, :]
    o_ptrs = O + bid_z * stride_oz + bid_h * stride_oh + offs_m[:, None] * stride_om + tl.arange(0, D_HEAD)[None, :]
    acc_ptrs = ACC + bid_z * stride_accz + bid_h * stride_acch + offs_m[:, None] * stride_accm + offs_n[None, :] * stride_accn

    q_mask = (offs_m < N_CTX)[:, None] & (offs_n < D_HEAD)[None, :]
    k_mask = (offs_m < N_CTX)[:, None] & (offs_n < D_HEAD)[None, :]
    valid_m = offs_m < N_CTX
    if STAGE == 0:
        q_data = tl.where(q_mask, tl.load(q_ptrs, mask=q_mask, other=0.), 0.)
        k_data = tl.where(k_mask, tl.load(k_ptrs, mask=k_mask, other=0.), 0.)
        q_data = q_data * QSCALE
        k_data = k_data * KSCALE
        dot = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for d in range(0, D_HEAD):
            qv = q_data[:, d]
            kv = k_data[:, d]
            dot += qv[:, None] * kv[None, :]
        dot = tl.where(valid_m[:, None] & (offs_n < N_CTX)[None, :], dot, float("-inf"))
        tl.store(acc_ptrs, dot)
    if STAGE == 1:
        scores = tl.load(acc_ptrs)
        scores = scores * SCALE
        exp_scores = tl.exp(scores)
        denom = tl.sum(exp_scores, 1)
        tl.store(acc_ptrs, exp_scores)
        tl.atomic_xchg(acc_ptrs + BLOCK_N * BLOCK_M, denom)
    if STAGE == 2:
        exp_scores = tl.load(acc_ptrs)
        denom = tl.load(acc_ptrs + BLOCK_N * BLOCK_M)
        softmax = exp_scores / denom[:, None]
        acc_o = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
        for nn in range(0, BLOCK_N):
            mask_n = offs_n + nn < N_CTX
        for d in range(0, D_HEAD):
            w = softmax[:, nn]
            val = tl.load(v_ptrs + nn * stride_vn + d, mask=mask_n, other=0.)
            acc_o[:, d] += w * val
        tl.store(o_ptrs, acc_o, mask=valid_m[:, None])


@triton.jit
def _attn_fwd(
    Q, K, V, O,
    ACC, QSCALE, KSCALE,
    stride_qz, stride_qh, stride_qm, stride_qq,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_oo,
    stride_accz, stride_acch, stride_accm, stride_accn,
    B, H, N_CTX, D_HEAD,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    grid_z = tl.launch.grid(0)
    grid_h = tl.launch.grid(1)
    grid_m = tl.launch.grid(2)
    _attn_fwd_inner[grid_z, grid_h, grid_m](
        Q, K, V, O, ACC, 1.0, QSCALE, KSCALE,
        stride_qz, stride_qh, stride_qm, stride_qq,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_oo,
        stride_accz, stride_acch, stride_accm, stride_accn,
        B, H, N_CTX, D_HEAD, BLOCK_M, BLOCK_N, STAGE=0
    )
    _attn_fwd_inner[grid_z, grid_h, grid_m](
        Q, K, V, O, ACC, 1.0, QSCALE, KSCALE,
        stride_qz, stride_qh, stride_qm, stride_qq,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_oo,
        stride_accz, stride_acch, stride_accm, stride_accn,
        B, H, N_CTX, D_HEAD, BLOCK_M, BLOCK_N, STAGE=1
    )
    _attn_fwd_inner[grid_z, grid_h, grid_m](
        Q, K, V, O, ACC, 1.0, QSCALE, KSCALE,
        stride_qz, stride_qh, stride_qm, stride_qq,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_oo,
        stride_accz, stride_acch, stride_accm, stride_accn,
        B, H, N_CTX, D_HEAD, BLOCK_M, BLOCK_N, STAGE=2
    )


def forward(q, k, v, q_scale, k_scale):
    import torch
    B, H, N_CTX, D_HEAD = q.shape
    o = torch.empty_like(q)
    acc = torch.zeros((B, H, N_CTX, N_CTX + 1), dtype=q.dtype, device=q.device)
    grid = (B, H, (N_CTX + 63) // 64)
    _attn_fwd[grid](
        q, k, v, o, acc,
        q_scale, k_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        B, H, N_CTX, D_HEAD,
        BLOCK_M=64,
        BLOCK_N=64
    )
    return o
