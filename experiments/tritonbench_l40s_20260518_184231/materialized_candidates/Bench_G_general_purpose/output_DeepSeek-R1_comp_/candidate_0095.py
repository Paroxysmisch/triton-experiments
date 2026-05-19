import torch
import triton
import triton.language as tl

@triton.jit
def fused_recurrent_retention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, initial_state_ptr,
    o_ptr, h_final_ptr,
    scale,
    batch_size, num_heads, seq_len, dim_k, dim_v,
    stride_q_b, stride_q_h, stride_q_t, stride_q_d,
    stride_k_b, stride_k_h, stride_k_t, stride_k_d,
    stride_v_b, stride_v_h, stride_v_t, stride_v_d,
    stride_o_b, stride_o_h, stride_o_t, stride_o_d,
    stride_h_b, stride_h_h, stride_h_k, stride_h_v,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_k = tl.program_id(1)
    i_bh = tl.program_id(2)
    
    batch = i_bh // num_heads
    head = i_bh % num_heads
    
    offs_k = i_k * BK + tl.arange(0, BK)
    offs_v = i_v * BV + tl.arange(0, BV)
    offs_b = batch
    offs_h = head
    
    if USE_INITIAL_STATE:
        h_ptr = initial_state_ptr + offs_b * stride_h_b + offs_h * stride_h_h + offs_k[:, None] * stride_h_k + offs_v[None, :] * stride_h_v
        h = tl.load(h_ptr, mask=(offs_k[:, None] < dim_k) & (offs_v[None, :] < dim_v), other=0.0)
    else:
        h = tl.zeros((BK, BV), dtype=tl.float32)
    
    gamma = 1.0 / (2 ** (head + 1))
    
    for t in range(seq_len):
        q_ptr_t = q_ptr + offs_b * stride_q_b + offs_h * stride_q_h + t * stride_q_t + offs_k
        k_ptr_t = k_ptr + offs_b * stride_k_b + offs_h * stride_k_h + t * stride_k_t + offs_k
        v_ptr_t = v_ptr + offs_b * stride_v_b + offs_h * stride_v_h + t * stride_v_t + offs_v
        
        mask_k = offs_k < dim_k
        mask_v = offs_v < dim_v
        
        q = tl.load(q_ptr_t, mask=mask_k, other=0.0)
        k = tl.load(k_ptr_t, mask=mask_k, other=0.0)
        v = tl.load(v_ptr_t, mask=mask_v, other=0.0)
        
        outer = k[:, None] * v[None, :]
        h = h * gamma + outer
        
        partial = tl.sum(q[:, None] * h, axis=0)
        partial *= scale
        
        o_ptr_t = o_ptr + offs_b * stride_o_b + offs_h * stride_o_h + t * stride_o_t + offs_v
        tl.atomic_add(o_ptr_t, partial, mask=mask_v)
    
    if STORE_FINAL_STATE:
        h_final_ptr = h_final_ptr + offs_b * stride_h_b + offs_h * stride_h_h + offs_k[:, None] * stride_h_k + offs_v[None, :] * stride_h_v
        tl.store(h_final_ptr, h, mask=(offs_k[:, None] < dim_k) & (offs_v[None, :] < dim_v))

@triton.jit
def fused_recurrent_retention_bwd_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, initial_state_ptr,
    dq_ptr, dk_ptr, dv_ptr, dh_initial_ptr,
    scale,
    batch_size, num_heads, seq_len, dim_k, dim_v,
    stride_q_b, stride_q_h, stride_q_t, stride_q_d,
    stride_k_b, stride_k_h, stride_k_t, stride_k_d,
    stride_v_b, stride_v_h, stride_v_t, stride_v_d,
    stride_do_b, stride_do_h, stride_do_t, stride_do_d,
    stride_dq_b, stride_dq_h, stride_dq_t, stride_dq_d,
    stride_dk_b, stride_dk_h, stride_dk_t, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_t, stride_dv_d,
    stride_h_b, stride_h_h, stride_h_k, stride_h_v,
    USE_INITIAL_STATE: tl.constexpr,
    HAS_DH_INITIAL: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_k = tl.program_id(1)
    i_bh = tl.program_id(2)
    
    batch = i_bh // num_heads
    head = i_bh % num_heads
    
    offs_k = i_k * BK + tl.arange(0, BK)
    offs_v = i_v * BV + tl.arange(0, BV)
    offs_b = batch
    offs_h = head
    
    if HAS_DH_INITIAL:
        dh_ptr = dh_initial_ptr + offs_b * stride_h_b + offs_h * stride_h_h + offs_k[:, None] * stride_h_k + offs_v[None, :] * stride_h_v
        dh = tl.load(dh_ptr, mask=(offs_k[:, None] < dim_k) & (offs_v[None, :] < dim_v), other=0.0)
    else:
        dh = tl.zeros((BK, BV), dtype=tl.float32)
    
    gamma = 1.0 / (2 ** (head + 1))
    
    for t in range(seq_len - 1, -1, -1):
        q_ptr_t = q_ptr + offs_b * stride_q_b + offs_h * stride_q_h + t * stride_q_t + offs_k
        k_ptr_t = k_ptr + offs_b * stride_k_b + offs_h * stride_k_h + t * stride_k_t + offs_k
        v_ptr_t = v_ptr + offs_b * stride_v_b + offs_h * stride_v_h + t * stride_v_t + offs_v
        do_ptr_t = do_ptr + offs_b * stride_do_b + offs_h * stride_do_h + t * stride_do_t + offs_v
        
        mask_k = offs_k < dim_k
        mask_v = offs_v < dim_v
        
        q = tl.load(q_ptr_t, mask=mask_k, other=0.0)
        k = tl.load(k_ptr_t, mask=mask_k, other=0.0)
        v = tl.load(v_ptr_t, mask=mask_v, other=0.0)
        do = tl.load(do_ptr_t, mask=mask_v, other=0.0)
        
        dq_partial = tl.sum(dh * do[None, :] * scale, axis=1)
        dq_ptr_t = dq_ptr + offs_b * stride_dq_b + offs_h * stride_dq_h + t * stride_dq_t + offs_k
        tl.atomic_add(dq_ptr_t, dq_partial, mask=mask_k)
        
        dk_partial = tl.sum((dh * gamma) * q[:, None], axis=1)
        dk_ptr_t = dk_ptr + offs_b * stride_dk_b + offs_h * stride_dk_h + t * stride_dk_t + offs_k
        tl.store(dk_ptr_t, dk_partial, mask=mask_k)
        
        dv_partial = tl.sum((dh * gamma) * q[:, None], axis=0)
        dv_ptr_t = dv_ptr + offs_b * stride_dv_b + offs_h * stride_dv_h + t * stride_dv_t + offs_v
        tl.store(dv_ptr_t, dv_partial, mask=mask_v)
        
        dh = dh * gamma + k[:, None] * do[None, :] * scale

def fused_recurrent_retention(q, k, v, initial_state=None):
    batch, heads, seq_len, dim_k = q.shape
    _, _, _, dim_v = v.shape
    scale = dim_k ** -0.5
    
    o = torch.zeros_like(v)
    final_state = torch.zeros(batch, heads, dim_k, dim_v, device=q.device) if initial_state is not None else None
    
    BK, BV = 32, 32
    grid = (triton.cdiv(dim_v, BV), triton.cdiv(dim_k, BK), batch * heads)
    
    fused_recurrent_retention_fwd_kernel[grid](
        q, k, v, initial_state,
        o, final_state,
        scale,
        batch, heads, seq_len, dim_k, dim_v,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        final_state.stride(0) if final_state is not None else 0,
        final_state.stride(1) if final_state is not None else 0,
        final_state.stride(2) if final_state is not None else 0,
        final_state.stride(3) if final_state is not None else 0,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        BK=BK, BV=BV,
    )
    return o, final_state
