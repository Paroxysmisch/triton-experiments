import triton
import triton.language as tl

@triton.jit
def fwd_decay_cumsum(
    g_ptr, g_o_ptr,
    decay,
    inv_ln2,
    T, DK,
    stride_g_b, stride_g_h, stride_g_t, stride_g_k,
    stride_g_o_b, stride_g_o_h, stride_g_o_t, stride_g_o_k,
    BLOCK_T: tl.constexpr, BLOCK_DK: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_dk = tl.program_id(2)
    
    off_bh = pid_bh
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_dk = pid_dk * BLOCK_DK + tl.arange(0, BLOCK_DK)
    
    mask_t = off_t < T
    mask_dk = off_dk < DK
    mask = mask_t[:, None] & mask_dk[None, :]
    
    cum_decay = tl.zeros((BLOCK_T, BLOCK_DK), dtype=tl.float32)
    
    for i in tl.static_range(0, BLOCK_T):
        t = off_t[i]
        g_val = tl.load(g_ptr + off_bh * stride_g_b + t * stride_g_t + off_dk * stride_g_k,
                        mask=mask[i], other=0.0)
        scaled_g = g_val * inv_ln2
        if i > 0:
            cum_decay_prev = cum_decay[i-1, :] * decay
        else:
            cum_decay_prev = 0.0
        current_cum = cum_decay_prev + scaled_g
        cum_decay = tl.where(tl.arange(0, BLOCK_T)[:, None] == i, current_cum, cum_decay)
    
    tl.store(g_o_ptr + off_bh * stride_g_o_b + off_t[:, None] * stride_g_o_t + off_dk * stride_g_o_k,
             cum_decay, mask=mask)

def fwd_decay_cumsum_launch(g, g_o, decay, inv_ln2, BK=64, BT=128):
    B, H, T, DK = g.shape
    grid = (B * H, triton.cdiv(T, BT), triton.cdiv(DK, BK))
    fwd_decay_cumsum[grid](
        g, g_o, decay, inv_ln2, T, DK,
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        g_o.stride(0), g_o.stride(1), g_o.stride(2), g_o.stride(3),
        BLOCK_T=BT, BLOCK_DK=BK
    )

@triton.jit
def prepare_qg_kg(
    q_ptr, k_ptr, g_ptr,
    qg_ptr, kg_ptr,
    scale,
    decay,
    inv_ln2,
    T, DK,
    stride_q_b, stride_q_h, stride_q_t, stride_q_k,
    stride_k_b, stride_k_h, stride_k_t, stride_k_k,
    stride_g_b, stride_g_h, stride_g_t, stride_g_k,
    stride_qg_b, stride_qg_h, stride_qg_t, stride_qg_k,
    stride_kg_b, stride_kg_h, stride_kg_t, stride_kg_k,
    BLOCK_T: tl.constexpr, BLOCK_DK: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_dk = tl.program_id(2)
    
    off_bh = pid_bh
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_dk = pid_dk * BLOCK_DK + tl.arange(0, BLOCK_DK)
    
    mask_t = off_t < T
    mask_dk = off_dk < DK
    mask = mask_t[:, None] & mask_dk[None, :]
    
    cum_decay = tl.zeros((BLOCK_T, BLOCK_DK), dtype=tl.float32)
    
    for i in tl.static_range(0, BLOCK_T):
        t = off_t[i]
        g_val = tl.load(g_ptr + off_bh * stride_g_b + t * stride_g_t + off_dk * stride_g_k,
                        mask=mask[i], other=0.0)
        scaled_g = g_val * inv_ln2
        if i > 0:
            cum_decay_prev = cum_decay[i-1, :] * decay
        else:
            cum_decay_prev = 0.0
        current_cum = cum_decay_prev + scaled_g
        cum_decay = tl.where(tl.arange(0, BLOCK_T)[:, None] == i, current_cum, cum_decay)
        
        q_val = tl.load(q_ptr + off_bh * stride_q_b + t * stride_q_t + off_dk * stride_q_k,
                        mask=mask[i], other=0.0)
        k_val = tl.load(k_ptr + off_bh * stride_k_b + t * stride_k_t + off_dk * stride_k_k,
                        mask=mask[i], other=0.0)
        
        decay_factor = tl.exp2(-current_cum * scale)
        qg_val = q_val * decay_factor
        kg_val = k_val * decay_factor
        
        tl.store(qg_ptr + off_bh * stride_qg_b + t * stride_qg_t + off_dk * stride_qg_k,
                qg_val, mask=mask[i])
        tl.store(kg_ptr + off_bh * stride_kg_b + t * stride_kg_t + off_dk * stride_kg_k,
                kg_val, mask=mask[i])

def prepare_qg_kg_launch(q, k, g, qg, kg, scale, decay, inv_ln2, BK=64, BT=128):
    B, H, T, DK = q.shape
    grid = (B * H, triton.cdiv(T, BT), triton.cdiv(DK, BK))
    prepare_qg_kg[grid](
        q, k, g, qg, kg,
        scale, decay, inv_ln2, T, DK,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        qg.stride(0), qg.stride(1), qg.stride(2), qg.stride(3),
        kg.stride(0), kg.stride(1), kg.stride(2), kg.stride(3),
        BLOCK_T=BT, BLOCK_DK=BK
    )

@triton.jit
def bwd_decay_global_cumsum(
    dq_in_ptr, dq_out_ptr, dk_in_ptr, dk_out_ptr, g_ptr, dg_ptr,
    scale, decay, inv_ln2,
    T, DK,
    stride_dq_b, stride_dq_h, stride_dq_t, stride_dq_k,
    stride_dk_b, stride_dk_h, stride_dk_t, stride_dk_k,
    stride_g_b, stride_g_h, stride_g_t, stride_g_k,
    stride_dg_b, stride_dg_h, stride_dg_t, stride_dg_k,
    BLOCK_T: tl.constexpr, BLOCK_DK: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_dk = tl.program_id(2)
    
    off_bh = pid_bh
    off_t = (T // BLOCK_T - 1 - pid_t) * BLOCK_T + tl.arange(0, BLOCK_T)
    off_dk = pid_dk * BLOCK_DK + tl.arange(0, BLOCK_DK)
    
    mask_t = off_t < T
    mask_dk = off_dk < DK
    mask = mask_t[:, None] & mask_dk[None, :]
    
    carry_dg = tl.zeros((BLOCK_DK,), dtype=tl.float32)
    
    for i in tl.static_range(0, BLOCK_T):
        t = off_t[i]
        dq_val = tl.load(dq_in_ptr + off_bh * stride_dq_b + t * stride_dq_t + off_dk * stride_dq_k,
                         mask=mask[i], other=0.0)
        dk_val = tl.load(dk_in_ptr + off_bh * stride_dk_b + t * stride_dk_t + off_dk * stride_dk_k,
                         mask=mask[i], other=0.0)
        g_val = tl.load(g_ptr + off_bh * stride_g_b + t * stride_g_t + off_dk * stride_g_k,
                       mask=mask[i], other=0.0)
        
        scaled_g = g_val * inv_ln2
        decay_factor = tl.exp2(-scaled_g * scale)
        
        dg_part = (dq_val * decay_factor + dk_val * decay_factor) * (-scale * inv_ln2)
        carry_dg = carry_dg * decay + dg_part
        
        tl.store(dg_ptr + off_bh * stride_dg_b + t * stride_dg_t + off_dk * stride_dg_k,
                 carry_dg, mask=mask[i])
    
    tl.store(dq_out_ptr + off_bh * stride_dq_b + off_t[:, None] * stride_dq_t + off_dk * stride_dq_k,
            dq_val, mask=mask)
    tl.store(dk_out_ptr + off_bh * stride_dk_b + off_t[:, None] * stride_dk_t + off_dk * stride_dk_k,
            dk_val, mask=mask)

def bwd_decay_global_cumsum_launch(dq_in, dq_out, dk_in, dk_out, g, dg, scale, decay, inv_ln2, BK=64, BT=128):
    B, H, T, DK = dq_in.shape
    grid = (B * H, triton.cdiv(T, BT), triton.cdiv(DK, BK))
    bwd_decay_global_cumsum[grid](
        dq_in, dq_out, dk_in, dk_out, g, dg,
        scale, decay, inv_ln2, T, DK,
        dq_in.stride(0), dq_in.stride(1), dq_in.stride(2), dq_in.stride(3),
        dk_in.stride(0), dk_in.stride(1), dk_in.stride(2), dk_in.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        dg.stride(0), dg.stride(1), dg.stride(2), dg.stride(3),
        BLOCK_T=BT, BLOCK_DK=BK
    )
