import torch
import triton
import triton.language as tl

@triton.jit
def chunk_retention_fwd_kernel_h(
    k_ptr, v_ptr, h_ptr, h0_ptr, ht_ptr,
    BT: tl.constexpr, BC: tl.constexpr, F: tl.constexpr,
    scale: tl.constexpr, use_h0: tl.constexpr,
    stride_kb, stride_kc, stride_kt, stride_kf,
    stride_vb, stride_vc, stride_vt, stride_vf,
    stride_hb, stride_hc, stride_ht, stride_hf,
    BLOCK: tl.constexpr
):
    pid_bc = tl.program_id(0)
    pid_f = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask_f = pid_f < F

    off_bc = pid_bc
    off_c = off_bc % BC
    off_b = off_bc // BC

    h_prev = tl.zeros([BLOCK], dtype=tl.float32)
    if use_h0:
        h0_ptr += off_b * stride_hb + off_c * stride_hc + pid_f * stride_hf
        h_prev = tl.load(h0_ptr, mask=mask_f, other=0.0)

    for t in range(BT):
        k_ptr += off_b * stride_kb + off_c * stride_kc + t * stride_kt + pid_f * stride_kf
        v_ptr += off_b * stride_vb + off_c * stride_vc + t * stride_vt + pid_f * stride_vf
        h_ptr += off_b * stride_hb + off_c * stride_hc + t * stride_ht + pid_f * stride_hf
        
        k = tl.load(k_ptr, mask=mask_f, other=0.0)
        v = tl.load(v_ptr, mask=mask_f, other=0.0)
        
        h_cur = h_prev * (1 - k) + v * k
        h_prev = h_cur
        
        tl.store(h_ptr, h_cur, mask=mask_f)

    if ht_ptr != 0:
        ht_ptr += off_b * stride_hb + off_c * stride_hc + pid_f * stride_hf
        tl.store(ht_ptr, h_prev, mask=mask_f)

@triton.jit
def chunk_retention_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, o_ptr,
    BT: tl.constexpr, BC: tl.constexpr, F: tl.constexpr,
    scale: tl.constexpr,
    stride_qb, stride_qc, stride_qt, stride_qf,
    stride_kb, stride_kc, stride_kt, stride_kf,
    stride_vb, stride_vc, stride_vt, stride_vf,
    stride_hb, stride_hc, stride_ht, stride_hf,
    stride_ob, stride_oc, stride_ot, stride_of,
    BLOCK: tl.constexpr
):
    pid_t = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_f = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask_f = pid_f < F

    off_bc = pid_bc
    off_c = off_bc % BC
    off_b = off_bc // BC
    t = pid_t

    q_ptr += off_b * stride_qb + off_c * stride_qc + t * stride_qt + pid_f * stride_qf
    k_ptr += off_b * stride_kb + off_c * stride_kc + t * stride_kt + pid_f * stride_kf
    h_ptr += off_b * stride_hb + off_c * stride_hc + t * stride_ht + pid_f * stride_hf
    v_ptr += off_b * stride_vb + off_c * stride_vc + t * stride_vt + pid_f * stride_vf
    o_ptr += off_b * stride_ob + off_c * stride_oc + t * stride_ot + pid_f * stride_of

    q = tl.load(q_ptr, mask=mask_f, other=0.0)
    k = tl.load(k_ptr, mask=mask_f, other=0.0)
    h = tl.load(h_ptr, mask=mask_f, other=0.0)
    v = tl.load(v_ptr, mask=mask_f, other=0.0)

    o = q * h * scale + k * v * scale
    tl.store(o_ptr, o, mask=mask_f)

@triton.jit
def chunk_retention_bwd_kernel_dh(
    q_ptr, do_ptr, dh_ptr,
    BT: tl.constexpr, BC: tl.constexpr, F: tl.constexpr,
    scale: tl.constexpr,
    stride_qb, stride_qc, stride_qt, stride_qf,
    stride_ob, stride_oc, stride_ot, stride_of,
    stride_dhb, stride_dhc, stride_dht, stride_dhf,
    BLOCK: tl.constexpr
):
    pid_bc = tl.program_id(0)
    pid_f = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask_f = pid_f < F

    off_bc = pid_bc
    off_c = off_bc % BC
    off_b = off_bc // BC

    dh_prev = tl.zeros([BLOCK], dtype=tl.float32)
    for t in range(BT-1, -1, -1):
        q_ptr += off_b * stride_qb + off_c * stride_qc + t * stride_qt + pid_f * stride_qf
        do_ptr += off_b * stride_ob + off_c * stride_oc + t * stride_ot + pid_f * stride_of
        dh_ptr += off_b * stride_dhb + off_c * stride_dhc + t * stride_dht + pid_f * stride_dhf

        q = tl.load(q_ptr, mask=mask_f, other=0.0)
        do = tl.load(do_ptr, mask=mask_f, other=0.0)
        
        dh = dh_prev + q * do * scale
        dh_prev = dh
        
        tl.store(dh_ptr, dh, mask=mask_f)

@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    q_ptr, k_ptr, v_ptr, h_ptr, do_ptr, dh_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    BT: tl.constexpr, BC: tl.constexpr, F: tl.constexpr,
    scale: tl.constexpr,
    stride_qb, stride_qc, stride_qt, stride_qf,
    stride_kb, stride_kc, stride_kt, stride_kf,
    stride_vb, stride_vc, stride_vt, stride_vf,
    stride_hb, stride_hc, stride_ht, stride_hf,
    stride_ob, stride_oc, stride_ot, stride_of,
    stride_dqb, stride_dqc, stride_dqt, stride_dqf,
    stride_dkb, stride_dkc, stride_dkt, stride_dkf,
    stride_dvb, stride_dvc, stride_dvt, stride_dvf,
    BLOCK: tl.constexpr
):
    pid_t = tl.program_id(0)
    pid_bc = tl.program_id(1)
    pid_f = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask_f = pid_f < F

    off_bc = pid_bc
    off_c = off_bc % BC
    off_b = off_bc // BC
    t = pid_t

    # Load pointers
    q_ptr += off_b * stride_qb + off_c * stride_qc + t * stride_qt + pid_f * stride_qf
    k_ptr += off_b * stride_kb + off_c * stride_kc + t * stride_kt + pid_f * stride_kf
    v_ptr += off_b * stride_vb + off_c * stride_vc + t * stride_vt + pid_f * stride_vf
    h_ptr += off_b * stride_hb + off_c * stride_hc + t * stride_ht + pid_f * stride_hf
    do_ptr += off_b * stride_ob + off_c * stride_oc + t * stride_ot + pid_f * stride_of
    dh_ptr += off_b * stride_hb + off_c * stride_hc + t * stride_ht + pid_f * stride_hf

    # Load data
    q = tl.load(q_ptr, mask=mask_f)
    k = tl.load(k_ptr, mask=mask_f)
    v = tl.load(v_ptr, mask=mask_f)
    h = tl.load(h_ptr, mask=mask_f)
    do = tl.load(do_ptr, mask=mask_f)
    dh = tl.load(dh_ptr, mask=mask_f)

    # Compute gradients
    dq = tl.where(mask_f, dh * do * scale, 0.0)
    dk = tl.where(mask_f, do * q * scale * v - dh * h, 0.0)
    dv = tl.where(mask_f, do * q * scale * k, 0.0)

    # Atomic add to gradients
    tl.atomic_add(dq_ptr + q_ptr, dq)
    tl.atomic_add(dk_ptr + k_ptr, dk)
    tl.atomic_add(dv_ptr + v_ptr, dv)

def chunk_fwd_h_fn(k, v, h, h0, ht, BT, scale):
    BLOCK = 128
    B, C, T, F = k.shape
    BC = B * C
    grid = (BC, triton.cdiv(F, BLOCK))
    chunk_retention_fwd_kernel_h[grid](
        k, v, h, h0, ht, BT, BC, F, scale, h0 is not None,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        h.stride(0), h.stride(1), h.stride(2), h.stride(3),
        BLOCK=BLOCK
    )

def chunk_fwd_o_fn(q, k, v, h, o, BT, scale):
    BLOCK = 128
    B, C, T, F = q.shape
    BC = B * C
    grid = (T, BC, triton.cdiv(F, BLOCK))
    chunk_retention_fwd_kernel_o[grid](
        q, k, v, h, o, BT, BC, F, scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        h.stride(0), h.stride(1), h.stride(2), h.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        BLOCK=BLOCK
    )

def chunk_bwd_dh_fn(q, do, dh, BT, scale):
    BLOCK = 128
    B, C, T, F = q.shape
    BC = B * C
    grid = (BC, triton.cdiv(F, BLOCK))
    chunk_retention_bwd_kernel_dh[grid](
        q, do, dh, BT, BC, F, scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dh.stride(0), dh.stride(1), dh.stride(2), dh.stride(3),
        BLOCK=BLOCK
    )

def chunk_bwd_dqkv_fn(q, k, v, h, do, dh, dq, dk, dv, BT, scale):
    BLOCK = 128
    B, C, T, F = q.shape
    BC = B * C
    grid = (T, BC, triton.cdiv(F, BLOCK))
    chunk_retention_bwd_kernel_dqkv[grid](
        q, k, v, h, do, dh, dq, dk, dv,
        BT, BC, F, scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        h.stride(0), h.stride(1), h.stride(2), h.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        BLOCK=BLOCK
    )

class ChunkRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, h0, scale, BT, checkpointing):
        B, C, T, F = q.shape
        h = torch.empty_like(k)
        ht = torch.empty_like(h0) if h0 is not None else None
        
        chunk_fwd_h_fn(k, v, h, h0, ht, BT, scale)
        o = torch.empty_like(q)
        chunk_fwd_o_fn(q, k, v, h, o, BT, scale)
        
        ctx.save_for_backward(q, k, v, h, h0, ht)
        ctx.scale = scale
        ctx.BT = BT
        return o, ht

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, h, h0, ht = ctx.saved_tensors
        scale = ctx.scale
        BT = ctx.BT
        
        dh = torch.empty_like(h)
        chunk_bwd_dh_fn(q, do, dh, BT, scale)
        
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        chunk_bwd_dqkv_fn(q, k, v, h, do, dh, dq, dk, dv, BT, scale)
        
        return dq, dk, dv, None, None, None, None

def chunk_retention(q, k, v, initial_h=None, output_final_h=False, scale=None, checkpointing=False):
    scale = scale or (q.size(-1) ** -0.5)
    BT = 128  # Can be made configurable
    o, ht = ChunkRetentionFunction.apply(q, k, v, initial_h, scale, BT, checkpointing)
    return (o, ht) if output_final_h else o
