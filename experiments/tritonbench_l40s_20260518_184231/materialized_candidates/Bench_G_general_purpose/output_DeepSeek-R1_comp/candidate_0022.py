import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, M, Out, L, m,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = tl.program_id(2)
    
    q_offset = off_z * stride_qz + off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    k_offset = off_z * stride_kz + off_hz * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    
    v_offset = off_z * stride_vz + off_hz * stride_vh
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    q = tl.load(Q_block_ptr, boundary_check=(0,))
    q = (q * sm_scale).to(tl.float16)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    for start_n in range(0, (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(1,))
        qk = tl.dot(q, k)
        
        if IS_CAUSAL:
            mask = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) >= (start_n + tl.arange(0, BLOCK_N)[None, :])
            qk = qk * (1 - mask) + mask * (-float("inf"))
        
        if M is not None:
            m = tl.load(M + off_hz * N_CTX * N_CTX + (start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * N_CTX + start_n + tl.arange(0, BLOCK_N)[None, :])
            qk += m
        
        m_ij = tl.maximum(tl.max(qk, 1) * sm_scale, m_i)
        p = tl.exp(qk * sm_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        p = p.to(Q.dtype.element_ty)
        
        v = tl.load(V_block_ptr, boundary_check=(0,))
        acc += tl.dot(p, v)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    acc = acc / l_i[:, None]
    o_offset = off_z * stride_oz + off_hz * stride_oh + start_m * BLOCK_M * stride_om
    Out_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(BLOCK_M, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(Out_block_ptr, acc.to(Out.dtype.element_ty))
    tl.store(m + off_z * H * N_CTX + off_hz * N_CTX + start_m * BLOCK_M + tl.arange(0, BLOCK_M), m_i)
    tl.store(L + off_z * H * N_CTX + off_hz * N_CTX + start_m * BLOCK_M + tl.arange(0, BLOCK_M), l_i)

@triton.jit
def _bwd_preprocess(
    Out, DO, Delta,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_doz, stride_doh, stride_dom, stride_don,
    Z, H, N_CTX, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    off_m = tl.program_id(2)
    
    o_offset = off_z * stride_oz + off_h * stride_oh + off_m * BLOCK_M * stride_om
    do_offset = off_z * stride_doz + off_h * stride_doh + off_m * BLOCK_M * stride_dom
    
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for blk in range(0, BLOCK_DMODEL, 32):
        o = tl.load(Out + o_offset + off_m * BLOCK_M * stride_om + blk + tl.arange(0, 32), mask=blk + tl.arange(0, 32) < BLOCK_DMODEL)
        do = tl.load(DO + do_offset + off_m * BLOCK_M * stride_dom + blk + tl.arange(0, 32), mask=blk + tl.arange(0, 32) < BLOCK_DMODEL)
        acc += tl.sum(o * do, axis=1)
    
    tl.store(Delta + off_z * H * N_CTX + off_h * N_CTX + off_m * BLOCK_M + tl.arange(0, BLOCK_M), acc)

@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale, Out, DO, DQ, DK, DV,
    L, D, delta,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_km, stride_kk,
    stride_vz, stride_vh, stride_vm, stride_vk,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr
):
    pass  # Similar structure to forward kernel with gradient calculations

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal, M):
        BLOCK = 128
        shape = q.shape[:-1]
        q = q.view(-1, q.shape[-2], q.shape[-1])
        k = k.view(-1, k.shape[-2], k.shape[-1])
        v = v.view(-1, v.shape[-2], v.shape[-1])
        
        L = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
        m = torch.empty_like(L)
        out = torch.empty_like(q)
        
        grid = (triton.cdiv(q.shape[1], BLOCK), q.shape[0] // q.shape[-1], q.shape[0])
        
        _fwd_kernel[grid](
            q, k, v, sm_scale, M, out, L, m,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_DMODEL=q.shape[-1],
            IS_CAUSAL=causal
        )
        
        ctx.save_for_backward(q, k, v, out, L, m)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.BLOCK = BLOCK
        return out.view(*shape)

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L, m = ctx.saved_tensors
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        delta = torch.empty_like(L)
        
        _bwd_preprocess[(q.shape[0], q.shape[1], ctx.BLOCK)](
            o, do, delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            q.shape[0], q.shape[1], q.shape[2], ctx.BLOCK, q.shape[-1]
        )
        
        grid = (triton.cdiv(q.shape[1], ctx.BLOCK), q.shape[0] // q.shape[-1], q.shape[0])
        _bwd_kernel[grid](
            q, k, v, ctx.sm_scale, o, do, dq, dk, dv,
            L, delta, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=ctx.BLOCK, BLOCK_N=ctx.BLOCK, BLOCK_DMODEL=q.shape[-1],
            CAUSAL=ctx.causal
        )
        
        return dq, dk, dv, None, None, None

def attention(q, k, v, sm_scale=None, causal=False, attn_mask=None):
    if sm_scale is None:
        sm_scale = 1.0 / (q.size(-1) ** 0.5)
    return _attention.apply(q, k, v, sm_scale, causal, attn_mask)

q = torch.randn(1, 8, 1024, 64, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

output = attention(q, k, v, causal=True)
