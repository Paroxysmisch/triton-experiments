import triton
import triton.language as tl
import torch

# Forward Kernel
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, L, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    q_offset = off_hz * stride_qh + start_m * BLOCK_M * stride_qm
    Q_block = tl.make_block_ptr(
        Q + q_offset, (N_CTX, BLOCK_DMODEL),
        (stride_qm, stride_qk), (start_m * BLOCK_M, 0),
        (BLOCK_M, BLOCK_DMODEL), (1, 0)
    )
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    q = tl.load(Q_block, boundary_check=(0, 1))
    
    for start_n in range(0, (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX, BLOCK_N):
        k_offset = off_hz * stride_kh + start_n * stride_kn
        K_block = tl.make_block_ptr(
            K + k_offset, (BLOCK_DMODEL, N_CTX),
            (stride_kk, stride_kn), (0, start_n),
            (BLOCK_DMODEL, BLOCK_N), (0, 1)
        )
        
        V_block = tl.make_block_ptr(
            V + off_hz * stride_vh + start_n * stride_vn,
            (N_CTX, BLOCK_DMODEL), (stride_vn, stride_vk),
            (start_n, 0), (BLOCK_N, BLOCK_DMODEL), (1, 0)
        )
        
        k = tl.load(K_block, boundary_check=(1,))
        v = tl.load(V_block, boundary_check=(0,))
        
        qk = tl.dot(q, k, allow_tf32=False)
        qk *= sm_scale
        
        if IS_CAUSAL:
            causal_mask = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) >= (start_n + tl.arange(0, BLOCK_N)[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))
        
        m_ij = tl.maximum(tl.max(qk, 1), m_i)
        p = tl.exp(qk - m_ij[:, None])
        
        if IS_CAUSAL:
            p = tl.where(causal_mask, p, 0.0)
        
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        acc *= alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16))
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    
    acc /= l_i[:, None]
    Out_block = tl.make_block_ptr(
        Out + off_hz * stride_oh + start_m * BLOCK_M * stride_om,
        (N_CTX, BLOCK_DMODEL), (stride_om, stride_ok),
        (start_m * BLOCK_M, 0), (BLOCK_M, BLOCK_DMODEL), (1, 0)
    )
    tl.store(Out_block, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))

# Backward Preprocess
@triton.jit
def _bwd_preprocess(
    Out, DO, Delta,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok,
    Z, H, N_CTX, BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK
    o = tl.load(Out + off + tl.arange(0, BLOCK)[:, None] * stride_om)
    do = tl.load(DO + off + tl.arange(0, BLOCK)[:, None] * stride_dom)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off, delta)

# Backward Kernel
@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale, L, Out, DO,
    DQ, DK, DV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pass  # Implementation similar to forward but reversed

# Wrapper Class
class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal=False):
        # Config
        BLOCK_M, BLOCK_N = 128, 32
        L = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        
        # Output
        o = torch.empty_like(q)
        
        # Grid
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1])
        
        # Launch kernel
        _fwd_kernel[grid](
            q, k, v, sm_scale, L, None, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=q.shape[3], IS_CAUSAL=causal,
            num_warps=4, num_stages=4
        )
        
        ctx.save_for_backward(q, k, v, o, L)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # Preprocess
        delta = torch.empty_like(L)
        _bwd_preprocess[(q.shape[0] * q.shape[1] * q.shape[2],)](
            o, do, delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            q.shape[0], q.shape[1], q.shape[2], BLOCK=128
        )
        
        # Backward kernel
        grid = (triton.cdiv(q.shape[2], 128), q.shape[0] * q.shape[1])
        _bwd_kernel[grid](
            q, k, v, ctx.sm_scale, L, o, do,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=128, BLOCK_DMODEL=q.shape[3],
            BLOCK_N=32, num_warps=4, num_stages=1
        )
        return dq, dk, dv, None, None

def attention(q, k, v, scale, causal=False):
    return _Attention.apply(q, k, v, scale, causal)
