import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, M, Y,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    H, N_CTX,
    BLOCK: tl.constexpr, BLOCK_MODEL: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
    
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_MODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK, 0),
        block_shape=(BLOCK, BLOCK_MODEL),
        order=(1, 0)
    )
    
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_MODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_MODEL, BLOCK),
        order=(0, 1)
    )
    
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, BLOCK_MODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK, BLOCK_MODEL),
        order=(1, 0)
    )
    
    q = tl.load(Q_block_ptr, boundary_check=(0, 1))
    k = tl.load(K_block_ptr, boundary_check=(0, 1))
    v = tl.load(V_block_ptr, boundary_check=(0, 1))
    
    qk = tl.dot(q, k) * sm_scale
    qk = tl.where(start_m * BLOCK + tl.arange(0, BLOCK)[:, None] >= tl.arange(0, BLOCK)[None, :], qk, float("-inf"))
    
    m = tl.max(qk, 1)
    p = tl.exp(qk - m[:, None])
    l = tl.sum(p, 1)
    p /= l[:, None]
    
    y = tl.dot(p.to(Q.dtype.element_ty), v)
    
    tl.store(L + off_hz * N_CTX + start_m * BLOCK + tl.arange(0, BLOCK), l)
    tl.store(M + off_hz * N_CTX + start_m * BLOCK + tl.arange(0, BLOCK), m)
    
    Y_block_ptr = tl.make_block_ptr(
        base=Y + off_hz * stride_qh,
        shape=(N_CTX, BLOCK_MODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK, 0),
        block_shape=(BLOCK, BLOCK_MODEL),
        order=(1, 0)
    )
    tl.store(Y_block_ptr, y.to(Y.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def _bwd_intra_kernel(
    Q, K, V, sm_scale, DO,
    DQ, DK, DV,
    L, M,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    H, N_CTX,
    BLOCK: tl.constexpr, CBLOCK: tl.constexpr, BLOCK_MODEL: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N_CTX, CBLOCK)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m
    
    offs_m = pid_m * CBLOCK + tl.arange(0, CBLOCK)
    offs_n = pid_n * CBLOCK + tl.arange(0, CBLOCK)
    
    q = tl.load(Q + offs_m[:, None] * stride_qm + stride_qk * tl.arange(0, BLOCK_MODEL)[None, :])
    k = tl.load(K + offs_n[:, None] * stride_kn + stride_kk * tl.arange(0, BLOCK_MODEL)[None, :])
    v = tl.load(V + offs_n[:, None] * stride_kn + stride_kk * tl.arange(0, BLOCK_MODEL)[None, :])
    do = tl.load(DO + offs_m[:, None] * stride_qm + stride_qk * tl.arange(0, BLOCK_MODEL)[None, :])
    
    qk = tl.dot(q, tl.trans(k)) * sm_scale
    qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
    
    m = tl.load(M + offs_m)
    p = tl.exp(qk - m[:, None])
    l = tl.load(L + offs_m)
    p /= l[:, None]
    
    dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)))
    ds = p * (dp - tl.sum(p * dp, axis=1)[:, None]) * sm_scale
    
    dq = tl.dot(ds.to(q.dtype), k)
    dk = tl.dot(tl.trans(ds.to(k.dtype)), q)
    dv = tl.dot(tl.trans(p.to(v.dtype)), do)
    
    tl.store(DQ + offs_m[:, None] * stride_qm + stride_qk * tl.arange(0, BLOCK_MODEL)[None, :], dq)
    tl.store(DK + offs_n[:, None] * stride_kn + stride_kk * tl.arange(0, BLOCK_MODEL)[None, :], dk)
    tl.store(DV + offs_n[:, None] * stride_kn + stride_kk * tl.arange(0, BLOCK_MODEL)[None, :], dv)

@triton.jit
def _bwd_inter_kernel(
    Q, K, V, sm_scale, DO,
    DQ, DK, DV,
    L, M,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    H, N_CTX,
    BLOCK: tl.constexpr, BLOCK_MODEL: tl.constexpr
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    
    for start_n in range(0, N_CTX, BLOCK):
        start_m = start_n
        q = tl.load(Q + off_hz * stride_qh + start_m * stride_qm + tl.arange(0, BLOCK_MODEL)[None, :])
        k = tl.load(K + off_hz * stride_kh + start_n * stride_kn + tl.arange(0, BLOCK_MODEL)[None, :])
        v = tl.load(V + off_hz * stride_kh + start_n * stride_kn + tl.arange(0, BLOCK_MODEL)[None, :])
        
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where((start_m + tl.arange(0, BLOCK)[:, None]) >= (start_n + tl.arange(0, BLOCK)[None, :]), qk, float("-inf"))
        
        m = tl.load(M + off_hz * N_CTX + start_m + tl.arange(0, BLOCK))
        p = tl.exp(qk - m[:, None])
        l = tl.load(L + off_hz * N_CTX + start_m + tl.arange(0, BLOCK))
        p /= l[:, None]
        
        do = tl.load(DO + off_hz * stride_qh + start_m * stride_qm + tl.arange(0, BLOCK_MODEL)[None, :])
        dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)))
        ds = p * (dp - tl.sum(p * dp, axis=1)[:, None]) * sm_scale
        
        dk = tl.dot(tl.trans(ds.to(k.dtype)), q)
        dv = tl.dot(tl.trans(p.to(v.dtype)), do)
        
        tl.store(DK + off_hz * stride_kh + start_n * stride_kn + tl.arange(0, BLOCK_MODEL)[None, :], 
                tl.load(DK + off_hz * stride_kh + start_n * stride_kn + tl.arange(0, BLOCK_MODEL)[None, :]) + dk)
        tl.store(DV + off_hz * stride_kh + start_n * stride_kn + tl.arange(0, BLOCK_MODEL)[None, :], 
                tl.load(DV + off_hz * stride_kh + start_n * stride_kn + tl.arange(0, BLOCK_MODEL)[None, :]) + dv)

class LightningAttention2NoDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        BLOCK = 64
        BLOCK_MODEL = q.size(-1)
        assert BLOCK_MODEL in {16, 32, 64, 128}
        
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        M = torch.empty_like(L)
        y = torch.empty_like(q)
        
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1))
        _fwd_kernel[grid](
            q, k, v, sm_scale, L, M, y,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[1], q.shape[2],
            BLOCK=BLOCK, BLOCK_MODEL=BLOCK_MODEL
        )
        
        ctx.save_for_backward(q, k, v, L, M)
        ctx.sm_scale = sm_scale
        ctx.BLOCK = BLOCK
        ctx.BLOCK_MODEL = BLOCK_MODEL
        return y

    @staticmethod
    def backward(ctx, dy):
        q, k, v, L, M = ctx.saved_tensors
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        CBLOCK = 32
        
        # Intra-block backward
        grid_intra = (triton.cdiv(q.shape[2], CBLOCK) ** 2, q.shape[0] * q.shape[1))
        _bwd_intra_kernel[grid_intra](
            q, k, v, ctx.sm_scale, dy,
            dq, dk, dv, L, M,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            q.shape[1], q.shape[2],
            BLOCK=ctx.BLOCK, CBLOCK=CBLOCK, BLOCK_MODEL=ctx.BLOCK_MODEL
        )
        
        # Inter-block backward
        grid_inter = (q.shape[0] * q.shape[1),)
        _bwd_inter_kernel[grid_inter](
            q, k, v, ctx.sm_scale, dy,
            dq, dk, dv, L, M,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            q.shape[1], q.shape[2],
            BLOCK=ctx.BLOCK, BLOCK_MODEL=ctx.BLOCK_MODEL
        )
        
        return dq, dk, dv, None

def lightning_attention(q, k, v, sm_scale=1.0):
    return LightningAttention2NoDecay.apply(q, k, v, sm_scale)
