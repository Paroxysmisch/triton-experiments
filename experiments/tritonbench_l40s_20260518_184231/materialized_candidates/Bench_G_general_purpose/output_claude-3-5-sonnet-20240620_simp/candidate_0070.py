import torch
import triton
import triton.language as tl

@triton.jit
def chunk_retention_fwd_kernel_h(
    h_ptr, q_ptr, k_ptr, v_ptr, init_h_ptr,
    B, T, C, D,
    stride_hb, stride_ht, stride_hc,
    stride_qb, stride_qt, stride_qc,
    stride_kb, stride_kt, stride_kc,
    stride_vb, stride_vt, stride_vc,
    stride_ihb, stride_ihc,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // (T * C)
    t = (pid % (T * C)) // C
    c = pid % C

    h_ptr += b * stride_hb + t * stride_ht + c * stride_hc
    q_ptr += b * stride_qb + t * stride_qt + c * stride_qc
    k_ptr += b * stride_kb + t * stride_kt + c * stride_kc
    v_ptr += b * stride_vb + t * stride_vt + c * stride_vc
    init_h_ptr += b * stride_ihb + c * stride_ihc

    h = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if t == 0:
        h = tl.load(init_h_ptr + tl.arange(0, BLOCK_SIZE))
    else:
        h = tl.load(h_ptr - stride_ht + tl.arange(0, BLOCK_SIZE))

    q = tl.load(q_ptr + tl.arange(0, BLOCK_SIZE))
    k = tl.load(k_ptr + tl.arange(0, BLOCK_SIZE))
    v = tl.load(v_ptr + tl.arange(0, BLOCK_SIZE))

    h = h * tl.exp(-1.0) + q * k * v

    tl.store(h_ptr + tl.arange(0, BLOCK_SIZE), h)

@triton.jit
def chunk_retention_fwd_kernel_o(
    o_ptr, q_ptr, k_ptr, v_ptr, h_ptr,
    B, T, C, D,
    stride_ob, stride_ot, stride_oc,
    stride_qb, stride_qt, stride_qc,
    stride_kb, stride_kt, stride_kc,
    stride_vb, stride_vt, stride_vc,
    stride_hb, stride_ht, stride_hc,
    scale,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // (T * C)
    t = (pid % (T * C)) // C
    c = pid % C

    o_ptr += b * stride_ob + t * stride_ot + c * stride_oc
    q_ptr += b * stride_qb + t * stride_qt + c * stride_qc
    k_ptr += b * stride_kb + t * stride_kt + c * stride_kc
    v_ptr += b * stride_vb + t * stride_vt + c * stride_vc
    h_ptr += b * stride_hb + t * stride_ht + c * stride_hc

    q = tl.load(q_ptr + tl.arange(0, BLOCK_SIZE))
    k = tl.load(k_ptr + tl.arange(0, BLOCK_SIZE))
    v = tl.load(v_ptr + tl.arange(0, BLOCK_SIZE))
    h = tl.load(h_ptr + tl.arange(0, BLOCK_SIZE))

    o = scale * q * (k * v + h)

    tl.store(o_ptr + tl.arange(0, BLOCK_SIZE), o)

@triton.jit
def chunk_retention_bwd_kernel_dh(
    dh_ptr, do_ptr, q_ptr,
    B, T, C, D,
    stride_dhb, stride_dht, stride_dhc,
    stride_dob, stride_dot, stride_doc,
    stride_qb, stride_qt, stride_qc,
    scale,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // (T * C)
    t = (pid % (T * C)) // C
    c = pid % C

    dh_ptr += b * stride_dhb + t * stride_dht + c * stride_dhc
    do_ptr += b * stride_dob + t * stride_dot + c * stride_doc
    q_ptr += b * stride_qb + t * stride_qt + c * stride_qc

    do = tl.load(do_ptr + tl.arange(0, BLOCK_SIZE))
    q = tl.load(q_ptr + tl.arange(0, BLOCK_SIZE))

    dh = scale * do * q

    if t < T - 1:
        next_dh = tl.load(dh_ptr + stride_dht + tl.arange(0, BLOCK_SIZE))
        dh += next_dh * tl.exp(-1.0)

    tl.store(dh_ptr + tl.arange(0, BLOCK_SIZE), dh)

@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    dq_ptr, dk_ptr, dv_ptr,
    do_ptr, q_ptr, k_ptr, v_ptr, h_ptr,
    B, T, C, D,
    stride_dqb, stride_dqt, stride_dqc,
    stride_dkb, stride_dkt, stride_dkc,
    stride_dvb, stride_dvt, stride_dvc,
    stride_dob, stride_dot, stride_doc,
    stride_qb, stride_qt, stride_qc,
    stride_kb, stride_kt, stride_kc,
    stride_vb, stride_vt, stride_vc,
    stride_hb, stride_ht, stride_hc,
    scale,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    b = pid // (T * C)
    t = (pid % (T * C)) // C
    c = pid % C

    dq_ptr += b * stride_dqb + t * stride_dqt + c * stride_dqc
    dk_ptr += b * stride_dkb + t * stride_dkt + c * stride_dkc
    dv_ptr += b * stride_dvb + t * stride_dvt + c * stride_dvc
    do_ptr += b * stride_dob + t * stride_dot + c * stride_doc
    q_ptr += b * stride_qb + t * stride_qt + c * stride_qc
    k_ptr += b * stride_kb + t * stride_kt + c * stride_kc
    v_ptr += b * stride_vb + t * stride_vt + c * stride_vc
    h_ptr += b * stride_hb + t * stride_ht + c * stride_hc

    do = tl.load(do_ptr + tl.arange(0, BLOCK_SIZE))
    q = tl.load(q_ptr + tl.arange(0, BLOCK_SIZE))
    k = tl.load(k_ptr + tl.arange(0, BLOCK_SIZE))
    v = tl.load(v_ptr + tl.arange(0, BLOCK_SIZE))
    h = tl.load(h_ptr + tl.arange(0, BLOCK_SIZE))

    dq = scale * do * (k * v + h)
    dk = scale * do * q * v
    dv = scale * do * q * k

    tl.store(dq_ptr + tl.arange(0, BLOCK_SIZE), dq)
    tl.store(dk_ptr + tl.arange(0, BLOCK_SIZE), dk)
    tl.store(dv_ptr + tl.arange(0, BLOCK_SIZE), dv)

def chunk_fwd_h_fn(q, k, v, init_h):
    B, T, C, D = q.shape
    grid = (B * T * C,)
    chunk_retention_fwd_kernel_h[grid](
        q.contiguous(), k.contiguous(), v.contiguous(), init_h.contiguous(),
        B, T, C, D,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        init_h.stride(0), init_h.stride(1),
        BLOCK_SIZE=D
    )

def chunk_fwd_o_fn(q, k, v, h, scale):
    B, T, C, D = q.shape
    grid = (B * T * C,)
    o = torch.empty_like(q)
    chunk_retention_fwd_kernel_o[grid](
        o, q.contiguous(), k.contiguous(), v.contiguous(), h.contiguous(),
        B, T, C, D,
        o.stride(0), o.stride(1), o.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        scale,
        BLOCK_SIZE=D
    )
    return o

def chunk_bwd_dh_fn(do, q, scale):
    B, T, C, D = do.shape
    grid = (B * T * C,)
    dh = torch.empty_like(do)
    chunk_retention_bwd_kernel_dh[grid](
        dh, do.contiguous(), q.contiguous(),
        B, T, C, D,
        dh.stride(0), dh.stride(1), dh.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        scale,
        BLOCK_SIZE=D
    )
    return dh

def chunk_bwd_dqkv_fn(do, q, k, v, h, scale):
    B, T, C, D = do.shape
    grid = (B * T * C,)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    chunk_retention_bwd_kernel_dqkv[grid](
        dq, dk, dv,
        do.contiguous(), q.contiguous(), k.contiguous(), v.contiguous(), h.contiguous(),
        B, T, C, D,
        dq.stride(0), dq.stride(1), dq.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        scale,
        BLOCK_SIZE=D
    )
    return dq, dk, dv

class ChunkRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, init_state=None, scale=1.0):
        B, T, C, D = q.shape
        h = torch.zeros((B, T, C, D), device=q.device, dtype=q.dtype)
        if init_state is not None:
            h[:, 0] = init_state
        chunk_fwd_h_fn(q, k, v, h)
        o = chunk_fwd_o_fn(q, k, v, h, scale)
        ctx.save_for_backward(q, k, v, h)
        ctx.scale = scale
        return o, h[:, -1]

    @staticmethod
    def backward(ctx, do, d_last_state):
        q, k, v, h = ctx.saved_tensors
        scale = ctx.scale
        B, T, C, D = do.shape

        dh = torch.zeros_like(h)
        dh[:, -1] = d_last_state
        dh = chunk_bwd_dh_fn(do, q, scale)
        dq, dk, dv = chunk_bwd_dqkv_fn(do, q, k, v, h, scale)

        return dq, dk, dv, dh[:, 0], None

def chunk_retention(q, k, v, init_state=None, scale=1.0):
    return ChunkRetentionFunction.apply(q, k, v, init_state, scale)
