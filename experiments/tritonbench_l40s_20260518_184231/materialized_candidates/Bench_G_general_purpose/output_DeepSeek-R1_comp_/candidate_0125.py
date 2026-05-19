import torch
import triton
import triton.language as tl

@triton.jit
def parallel_rebased_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, z_ptr,
    B, H, T, S, D,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_zb, stride_zh, stride_zt,
    use_scale: tl.constexpr,
    use_normalize: tl.constexpr,
    BTL: tl.constexpr, BTS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    offs_t = pid_t * BTL + tl.arange(0, BTL)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(B, H, T, D),
        strides=(stride_qb, stride_qh, stride_qt, stride_qd),
        offsets=(pid_b, pid_h, pid_t * BTL, 0),
        block_shape=(1, BTL, BLOCK_D),
        order=(0, 1, 2, 3)
    )
    q = tl.load(q_block_ptr, boundary_check=(2,))

    acc = tl.zeros((BTL, D), dtype=tl.float32)
    m_i = tl.zeros((BTL,), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BTL,), dtype=tl.float32)

    num_s_blocks = tl.cdiv(S, BTS)
    for pid_s in range(num_s_blocks):
        k_block_ptr = tl.make_block_ptr(
            base=k_ptr,
            shape=(B, H, S, D),
            strides=(stride_kb, stride_kh, stride_ks, stride_kd),
            offsets=(pid_b, pid_h, pid_s * BTS, 0),
            block_shape=(1, BTS, BLOCK_D),
            order=(0, 1, 2, 3)
        )
        k = tl.load(k_block_ptr, boundary_check=(2,))

        v_block_ptr = tl.make_block_ptr(
            base=v_ptr,
            shape=(B, H, S, D),
            strides=(stride_vb, stride_vh, stride_vs, stride_vd),
            offsets=(pid_b, pid_h, pid_s * BTS, 0),
            block_shape=(1, BTS, BLOCK_D),
            order=(0, 1, 2, 3)
        )
        v = tl.load(v_block_ptr, boundary_check=(2,))

        s = tl.dot(q, tl.trans(k), allow_tf32=False)
        if use_scale:
            s = s * (1.0 / tl.sqrt(D))
        
        m_ij = tl.max(s, axis=1)
        p = tl.exp(s - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i = alpha * l_i + beta * l_ij
        m_i = m_i_new

        p_scaled = p / l_i[:, None]
        acc = acc * alpha[:, None] + beta[:, None] * tl.dot(p_scaled.to(v.dtype), v, allow_tf32=False)

    o_block_ptr = tl.make_block_ptr(
        base=o_ptr,
        shape=(B, H, T, D),
        strides=(stride_ob, stride_oh, stride_ot, stride_od),
        offsets=(pid_b, pid_h, pid_t * BTL, 0),
        block_shape=(1, BTL, BLOCK_D),
        order=(0, 1, 2, 3)
    )
    tl.store(o_block_ptr, acc.to(o_ptr.dtype.element_ty), boundary_check=(2,))

    z = tl.exp(m_i) * l_i
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(B, H, T),
        strides=(stride_zb, stride_zh, stride_zt),
        offsets=(pid_b, pid_h, pid_t * BTL),
        block_shape=(1, BTL),
        order=(0, 1, 2)
    )
    tl.store(z_block_ptr, z.to(z_ptr.dtype.element_ty), boundary_check=(2,))

@triton.jit
def _parallel_rebased_bwd_dq(
    dq_ptr, do_ptr, k_ptr, z_ptr,
    B, H, T, S, D,
    stride_dqb, stride_dqh, stride_dqt, stride_dqd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_zb, stride_zh, stride_zt,
    use_scale: tl.constexpr,
    BTL: tl.constexpr, BTS: tl.constexpr, BK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    offs_t = pid_t * BTL + tl.arange(0, BTL)
    offs_d = tl.arange(0, BLOCK_D)

    dq = tl.zeros((BTL, BLOCK_D), dtype=tl.float32)
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(B, H, T),
        strides=(stride_zb, stride_zh, stride_zt),
        offsets=(pid_b, pid_h, pid_t * BTL),
        block_shape=(1, BTL),
        order=(0, 1, 2)
    )
    z = tl.load(z_block_ptr, boundary_check=(2,))

    for pid_s in range(tl.cdiv(S, BTS)):
        k_block_ptr = tl.make_block_ptr(
            base=k_ptr,
            shape=(B, H, S, D),
            strides=(stride_kb, stride_kh, stride_ks, stride_kd),
            offsets=(pid_b, pid_h, pid_s * BTS, 0),
            block_shape=(1, BTS, BLOCK_D),
            order=(0, 1, 2, 3)
        )
        k = tl.load(k_block_ptr, boundary_check=(2,))

        do_block_ptr = tl.make_block_ptr(
            base=do_ptr,
            shape=(B, H, T, D),
            strides=(stride_ob, stride_oh, stride_ot, stride_od),
            offsets=(pid_b, pid_h, pid_t * BTL, 0),
            block_shape=(1, BTL, BLOCK_D),
            order=(0, 1, 2, 3)
        )
        do = tl.load(do_block_ptr, boundary_check=(2,))

        s = tl.dot(do, tl.trans(k), allow_tf32=False)
        if use_scale:
            s = s * (1.0 / tl.sqrt(D))
        
        p = s / z[:, None]
        dq += tl.dot(p, k, allow_tf32=False)

    dq_block_ptr = tl.make_block_ptr(
        base=dq_ptr,
        shape=(B, H, T, D),
        strides=(stride_dqb, stride_dqh, stride_dqt, stride_dqd),
        offsets=(pid_b, pid_h, pid_t * BTL, 0),
        block_shape=(1, BTL, BLOCK_D),
        order=(0, 1, 2, 3)
    )
    tl.store(dq_block_ptr, dq.to(dq_ptr.dtype.element_ty), boundary_check=(2,))

@triton.jit
def _parallel_rebased_bwd_dkv(
    dk_ptr, dv_ptr, q_ptr, do_ptr, z_ptr,
    B, H, T, S, D,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_zb, stride_zh, stride_zt,
    use_scale: tl.constexpr,
    BTL: tl.constexpr, BTS: tl.constexpr, BK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_s = pid_s * BTS + tl.arange(0, BTS)
    offs_d = tl.arange(0, BLOCK_D)

    dk = tl.zeros((BTS, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BTS, BLOCK_D), dtype=tl.float32)

    for pid_t in range(tl.cdiv(T, BTL)):
        q_block_ptr = tl.make_block_ptr(
            base=q_ptr,
            shape=(B, H, T, D),
            strides=(stride_qb, stride_qh, stride_qt, stride_qd),
            offsets=(pid_b, pid_h, pid_t * BTL, 0),
            block_shape=(1, BTL, BLOCK_D),
            order=(0, 1, 2, 3)
        )
        q = tl.load(q_block_ptr, boundary_check=(2,))

        do_block_ptr = tl.make_block_ptr(
            base=do_ptr,
            shape=(B, H, T, D),
            strides=(stride_ob, stride_oh, stride_ot, stride_od),
            offsets=(pid_b, pid_h, pid_t * BTL, 0),
            block_shape=(1, BTL, BLOCK_D),
            order=(0, 1, 2, 3)
        )
        do = tl.load(do_block_ptr, boundary_check=(2,))

        z_block_ptr = tl.make_block_ptr(
            base=z_ptr,
            shape=(B, H, T),
            strides=(stride_zb, stride_zh, stride_zt),
            offsets=(pid_b, pid_h, pid_t * BTL),
            block_shape=(1, BTL),
            order=(0, 1, 2)
        )
        z = tl.load(z_block_ptr, boundary_check=(2,))

        s_q = tl.dot(do, tl.trans(q), allow_tf32=False)
        if use_scale:
            s_q = s_q * (1.0 / tl.sqrt(D))
        
        p = s_q / z[None, :]
        dk += tl.dot(p, q, allow_tf32=False)
        dv += tl.dot(p.to(do.dtype), do, allow_tf32=False)

    dk_block_ptr = tl.make_block_ptr(
        base=dk_ptr,
        shape=(B, H, S, D),
        strides=(stride_kb, stride_kh, stride_ks, stride_kd),
        offsets=(pid_b, pid_h, pid_s * BTS, 0),
        block_shape=(1, BTS, BLOCK_D),
        order=(0, 1, 2, 3)
    )
    tl.store(dk_block_ptr, dk.to(dk_ptr.dtype.element_ty), boundary_check=(2,))

    dv_block_ptr = tl.make_block_ptr(
        base=dv_ptr,
        shape=(B, H, S, D),
        strides=(stride_vb, stride_vh, stride_vs, stride_vd),
        offsets=(pid_b, pid_h, pid_s * BTS, 0),
        block_shape=(1, BTS, BLOCK_D),
        order=(0, 1, 2, 3)
    )
    tl.store(dv_block_ptr, dv.to(dv_ptr.dtype.element_ty), boundary_check=(2,))

class ParallelBasedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, use_scale, use_normalize):
        B, H, T, D = q.shape
        _, _, S, _ = k.shape
        o = torch.empty_like(q)
        z = torch.empty(B, H, T, device=q.device, dtype=torch.float32)

        BLOCK_D = 128 if D <= 128 else 64
        BTL, BTS = 64, 64
        grid = (B, H, triton.cdiv(T, BTL))
        
        parallel_rebased_fwd_kernel[grid](
            q, k, v, o, z,
            B, H, T, S, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            z.stride(0), z.stride(1), z.stride(2),
            use_scale, use_normalize,
            BTL=BTL, BTS=BTS, BK=BLOCK_D, BV=BLOCK_D,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(q, k, v, z)
        ctx.use_scale = use_scale
        return o, z

    @staticmethod
    def backward(ctx, do, dz):
        q, k, v, z = ctx.saved_tensors
        B, H, T, D = q.shape
        _, _, S, _ = k.shape
        
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        BLOCK_D = 128 if D <= 128 else 64
        BTL, BTS = 64, 64
        
        grid_dq = (B, H, triton.cdiv(T, BTL))
        _parallel_rebased_bwd_dq[grid_dq](
            dq, do, k, z,
            B, H, T, S, D,
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            z.stride(0), z.stride(1), z.stride(2),
            ctx.use_scale,
            BTL=BTL, BTS=BTS, BK=BLOCK_D,
            BLOCK_D=BLOCK_D,
        )

        grid_dkv = (B, H, triton.cdiv(S, BTS))
        _parallel_rebased_bwd_dkv[grid_dkv](
            dk, dv, q, do, z,
            B, H, T, S, D,
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            z.stride(0), z.stride(1), z.stride(2),
            ctx.use_scale,
            BTL=BTL, BTS=BTS, BK=BLOCK_D,
            BLOCK_D=BLOCK_D,
        )

        return dq, dk, dv, None, None

def parallel_rebased(q, k, v, use_scale=True, use_normalize=True, return_both=False):
    assert q.size(-1) <= 128, "Feature dimension must be <= 128"
    B, H, T, D = q.shape
    S = k.size(2)
    
    o, z = ParallelBasedFunction.apply(q, k, v, use_scale, use_normalize)
    return (o, z) if return_both else o
