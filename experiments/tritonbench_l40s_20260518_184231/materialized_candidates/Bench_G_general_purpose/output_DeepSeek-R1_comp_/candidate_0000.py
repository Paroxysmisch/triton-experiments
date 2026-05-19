import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    B, H, D, scale,
    BLOCK: tl.constexpr,
    BLOCK_MODEL: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    num_pid_s = tl.num_programs(1)
    
    off_bh = pid_bh
    off_z = off_bh // H
    off_h = off_bh % H
    
    off_s = pid_s * BLOCK_MODEL
    range_q = off_s + tl.arange(0, BLOCK_MODEL)
    mask_q = range_q < B
    
    # Load Q block
    q = tl.load(Q + off_z * stride_qb + off_h * stride_qh + range_q[:, None] * stride_qd,
                mask=mask_q[:, None], other=0.0)
    
    # Initialize output
    o = tl.zeros((BLOCK_MODEL, D), dtype=tl.float32)
    
    # Compute intra-block attention
    for s in range(num_pid_s):
        off_k = s * BLOCK_MODEL
        range_k = off_k + tl.arange(0, BLOCK_MODEL)
        mask_k = range_k < B
        
        k = tl.load(K + off_z * stride_kb + off_h * stride_kh + range_k[:, None] * stride_kd,
                    mask=mask_k[:, None], other=0.0)
        v = tl.load(V + off_z * stride_vb + off_h * stride_vh + range_k[:, None] * stride_vd,
                    mask=mask_k[:, None], other=0.0)
        
        # Compute scores
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(mask_q[:, None] & mask_k[None, :], s, float('-inf'))
        p = tl.softmax(s, axis=1)
        
        o += tl.dot(p, v)
    
    # Write output
    tl.store(O + off_z * stride_ob + off_h * stride_oh + range_q[:, None] * stride_od,
             o.to(O.dtype.element_ty), mask=mask_q[:, None])

@triton.jit
def _bwd_intra_kernel(
    Q, K, V, DO, DQ, DK, DV,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_do, stride_dq, stride_dk, stride_dv,
    B, H, D, scale,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    off_bh = pid_bh
    off_z = off_bh // H
    off_h = off_bh % H
    
    off_c = pid_c * CBLOCK
    range_c = off_c + tl.arange(0, CBLOCK)
    mask_c = range_c < B
    
    # Load current chunk of Q, K, V, DO
    q = tl.load(Q + off_z * stride_qb + off_h * stride_qh + range_c[:, None] * stride_qd,
                mask=mask_c[:, None], other=0.0)
    k = tl.load(K + off_z * stride_kb + off_h * stride_kh + range_c[:, None] * stride_kd,
                mask=mask_c[:, None], other=0.0)
    v = tl.load(V + off_z * stride_vb + off_h * stride_vh + range_c[:, None] * stride_vd,
                mask=mask_c[:, None], other=0.0)
    do = tl.load(DO + off_z * stride_do + off_h * stride_do + range_c[:, None] * stride_do,
                 mask=mask_c[:, None], other=0.0)
    
    # Compute intra-block gradients
    dk = tl.zeros((CBLOCK, D), dtype=tl.float32)
    dv = tl.zeros((CBLOCK, D), dtype=tl.float32)
    
    for s in range(NUM_CBLOCK):
        off_s = s * CBLOCK
        range_s = off_s + tl.arange(0, CBLOCK)
        mask_s = range_s < B
        
        qs = tl.load(Q + off_z * stride_qb + off_h * stride_qh + range_s[:, None] * stride_qd,
                     mask=mask_s[:, None], other=0.0)
        dos = tl.load(DO + off_z * stride_do + off_h * stride_do + range_s[:, None] * stride_do,
                      mask=mask_s[:, None], other=0.0)
        
        # Compute attention
        s = tl.dot(qs, tl.trans(k)) * scale
        p = tl.softmax(s, axis=1)
        
        # Compute gradients
        dp = tl.dot(dos, tl.trans(v))
        ds = dp * p * (1 - p)
        dq_local = tl.dot(ds, k) * scale
        dk += tl.dot(tl.trans(ds), qs) * scale
        dv += tl.dot(tl.trans(p), dos)
    
    # Store gradients
    tl.store(DQ + off_z * stride_dq + off_h * stride_dq + range_c[:, None] * stride_dq,
            dq_local.to(DQ.dtype.element_ty), mask=mask_c[:, None])
    tl.store(DK + off_z * stride_dk + off_h * stride_dk + range_c[:, None] * stride_dk,
            dk.to(DK.dtype.element_ty), mask=mask_c[:, None])
    tl.store(DV + off_z * stride_dv + off_h * stride_dv + range_c[:, None] * stride_dv,
            dv.to(DV.dtype.element_ty), mask=mask_c[:, None])

@triton.jit
def _bwd_inter_kernel(
    DK, DV,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    B, H, D,
    BLOCK: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    off_z = pid_bh // H
    off_h = pid_bh % H
    
    for s in range(0, B, BLOCK):
        range_s = s + tl.arange(0, BLOCK)
        mask_s = range_s < B
        
        # Accumulate inter-block gradients
        dk = tl.load(DK + off_z * stride_kb + off_h * stride_kh + range_s[:, None] * stride_kd,
                     mask=mask_s[:, None], other=0.0)
        dv = tl.load(DV + off_z * stride_vb + off_h * stride_vh + range_s[:, None] * stride_vd,
                     mask=mask_s[:, None], other=0.0)
        
        # Update gradients (inter-block contributions)
        tl.store(DK + off_z * stride_kb + off_h * stride_kh + range_s[:, None] * stride_kd,
                dk, mask=mask_s[:, None])
        tl.store(DV + off_z * stride_vb + off_h * stride_vh + range_s[:, None] * stride_vd,
                dv, mask=mask_s[:, None])

class LightningAttention2NoDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        BLOCK = 64
        CBLOCK = 32
        scale = q.shape[-1] ** -0.5
        
        B, H, L, D = q.shape
        o = torch.empty_like(q)
        
        grid = (B * H, triton.cdiv(L, BLOCK))
        _fwd_kernel[grid](
            q, k, v, o,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            L, H, D, scale,
            BLOCK=BLOCK, BLOCK_MODEL=BLOCK
        )
        
        ctx.save_for_backward(q, k, v)
        ctx.BLOCK = BLOCK
        ctx.CBLOCK = CBLOCK
        return o

    @staticmethod
    def backward(ctx, do):
        BLOCK = ctx.BLOCK
        CBLOCK = ctx.CBLOCK
        q, k, v = ctx.saved_tensors
        scale = q.shape[-1] ** -0.5
        
        B, H, L, D = q.shape
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        # Intra-block backward
        grid_intra = (B * H, triton.cdiv(L, CBLOCK))
        _bwd_intra_kernel[grid_intra](
            q, k, v, do, dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            L, H, D, scale,
            CBLOCK=CBLOCK, NUM_CBLOCK=triton.cdiv(L, CBLOCK)
        )
        
        # Inter-block backward
        grid_inter = (B * H,)
        _bwd_inter_kernel[grid_inter](
            dk, dv,
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            L, H, D,
            BLOCK=BLOCK
        )
        
        return dq, dk, dv

# Wrapper function
def lightning_attention(q, k, v):
    return LightningAttention2NoDecay.apply(q, k, v)
