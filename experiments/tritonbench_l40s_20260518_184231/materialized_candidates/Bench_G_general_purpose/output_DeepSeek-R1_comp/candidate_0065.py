import torch
import triton
import triton.language as tl

@triton.jit
def fused_recurrent_fwd_kernel(
    q_ptr, k_ptr, v_ptr,
    beta_ptr, initial_state_ptr,
    output_ptr, final_state_ptr,
    scale,
    B, H, T, K, V,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    BETA_HAS_HEAD: tl.constexpr,
    USE_SCALE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_f = tl.program_id(2)
    
    feature_range = pid_f * BV + tl.arange(0, BV)
    v_mask = feature_range < V
    
    batch_stride = H * T * V
    head_stride = T * V
    time_stride = V
    
    off_batch = pid_b * batch_stride
    off_head = pid_h * head_stride
    
    if USE_INITIAL_STATE:
        initial_state = tl.load(initial_state_ptr + pid_b * H * V + pid_h * V + feature_range, mask=v_mask, other=0.0)
    else:
        initial_state = tl.zeros((BV,), dtype=tl.float32)
    
    current_state = initial_state
    for t in range(T):
        off_time = t * time_stride
        off = off_batch + off_head + off_time + feature_range
        
        q = tl.load(q_ptr + off, mask=v_mask, other=0.0)
        k = tl.load(k_ptr + off, mask=v_mask, other=0.0)
        v = tl.load(v_ptr + off, mask=v_mask, other=0.0)
        
        if USE_SCALE:
            product = scale * q * k * v
        else:
            product = q * k * v
        
        if BETA_HAS_HEAD:
            beta = tl.load(beta_ptr + pid_h)
        else:
            beta = tl.load(beta_ptr)
        
        current_state = beta * current_state + product
        tl.store(output_ptr + off, current_state, mask=v_mask)
    
    if final_state_ptr is not None:
        final_state_off = pid_b * H * V + pid_h * V + feature_range
        tl.store(final_state_ptr + final_state_off, current_state, mask=v_mask)

@triton.jit
def fused_recurrent_bwd_kernel(
    q_ptr, k_ptr, v_ptr,
    beta_ptr, initial_state_ptr, output_ptr,
    grad_output_ptr,
    d_q_ptr, d_k_ptr, d_v_ptr,
    d_beta_ptr, d_initial_state_ptr,
    scale,
    B, H, T, K, V,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    BETA_HAS_HEAD: tl.constexpr,
    USE_SCALE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_f = tl.program_id(2)
    
    feature_range = pid_f * BV + tl.arange(0, BV)
    v_mask = feature_range < V
    
    batch_stride = H * T * V
    head_stride = T * V
    time_stride = V
    
    off_batch = pid_b * batch_stride
    off_head = pid_h * head_stride
    
    d_h = tl.zeros((BV,), dtype=tl.float32)
    
    d_q_accum = tl.zeros((BV,), dtype=tl.float32)
    d_k_accum = tl.zeros((BV,), dtype=tl.float32)
    d_v_accum = tl.zeros((BV,), dtype=tl.float32)
    d_beta_accum = tl.zeros((1,), dtype=tl.float32) if BETA_HAS_HEAD else 0.0
    
    for t in range(T-1, -1, -1):
        off_time = t * time_stride
        off = off_batch + off_head + off_time + feature_range
        
        q = tl.load(q_ptr + off, mask=v_mask, other=0.0)
        k = tl.load(k_ptr + off, mask=v_mask, other=0.0)
        v = tl.load(v_ptr + off, mask=v_mask, other=0.0)
        grad_out = tl.load(grad_output_ptr + off, mask=v_mask, other=0.0) + d_h
        
        if USE_SCALE:
            scaled_grad = scale * grad_out
        else:
            scaled_grad = grad_out
        
        d_q = scaled_grad * k * v
        d_k = scaled_grad * q * v
        d_v = scaled_grad * q * k
        
        d_q_accum += d_q
        d_k_accum += d_k
        d_v_accum += d_v
        
        if BETA_HAS_HEAD:
            beta = tl.load(beta_ptr + pid_h)
        else:
            beta = tl.load(beta_ptr)
        
        if t > 0:
            h_prev = tl.load(output_ptr + off - time_stride, mask=v_mask, other=0.0)
        else:
            if USE_INITIAL_STATE:
                h_prev = tl.load(initial_state_ptr + pid_b * H * V + pid_h * V + feature_range, mask=v_mask, other=0.0)
            else:
                h_prev = tl.zeros((BV,), dtype=tl.float32)
        
        if BETA_HAS_HEAD:
            d_beta_accum += tl.sum(grad_out * h_prev)
        else:
            d_beta_accum = tl.sum(grad_out * h_prev)
        
        d_h = beta * grad_out
    
    tl.store(d_q_ptr + off_batch + off_head + feature_range, d_q_accum, mask=v_mask)
    tl.store(d_k_ptr + off_batch + off_head + feature_range, d_k_accum, mask=v_mask)
    tl.store(d_v_ptr + off_batch + off_head + feature_range, d_v_accum, mask=v_mask)
    
    if BETA_HAS_HEAD:
        tl.atomic_add(d_beta_ptr + pid_h, d_beta_accum)
    elif beta_ptr is not None:
        tl.atomic_add(d_beta_ptr, d_beta_accum)
    
    if USE_INITIAL_STATE and d_initial_state_ptr is not None:
        tl.store(d_initial_state_ptr + pid_b * H * V + pid_h * V + feature_range, d_h, mask=v_mask)

class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, initial_state, scale):
        B, H, T, K = q.shape
        V = v.shape[-1]
        output = torch.empty_like(v)
        final_state = None
        if initial_state is None:
            final_state = torch.empty((B, H, V), dtype=q.dtype, device=q.device)
        
        BK = min(triton.next_power_of_2(K), 64)
        BV = min(triton.next_power_of_2(V), 64)
        grid = (B, H, triton.cdiv(V, BV))
        
        fused_recurrent_fwd_kernel[grid](
            q, k, v,
            beta, initial_state,
            output, final_state,
            scale,
            B, H, T, K, V,
            BK, BV,
            USE_INITIAL_STATE=initial_state is not None,
            BETA_HAS_HEAD=beta is not None and beta.dim() == 1 and beta.size(0) == H,
            USE_SCALE=scale is not None
        )
        
        ctx.save_for_backward(q, k, v, beta, initial_state, output)
        ctx.scale = scale
        return output, final_state

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        q, k, v, beta, initial_state, output = ctx.saved_tensors
        scale = ctx.scale
        
        B, H, T, K = q.shape
        V = v.shape[-1]
        
        d_q = torch.zeros_like(q)
        d_k = torch.zeros_like(k)
        d_v = torch.zeros_like(v)
        d_beta = None
        d_initial_state = None
        
        if beta is not None:
            d_beta = torch.zeros_like(beta)
        if initial_state is not None:
            d_initial_state = torch.zeros_like(initial_state)
        
        BK = min(triton.next_power_of_2(K), 64)
        BV = min(triton.next_power_of_2(V), 64)
        grid = (B, H, triton.cdiv(V, BV))
        
        fused_recurrent_bwd_kernel[grid](
            q, k, v,
            beta, initial_state, output,
            grad_output,
            d_q, d_k, d_v,
            d_beta, d_initial_state,
            scale,
            B, H, T, K, V,
            BK, BV,
            USE_INITIAL_STATE=initial_state is not None,
            BETA_HAS_HEAD=beta is not None and beta.dim() == 1 and beta.size(0) == H,
            USE_SCALE=scale is not None
        )
        
        return d_q, d_k, d_v, d_beta, d_initial_state, None

def fused_recurrent_delta_rule(q, k, v, beta=None, initial_state=None, scale=None):
    if scale is None:
        scale = 1.0 / (q.size(-1) ** 0.5)
    output, final_state = FusedRecurrentFunction.apply(q, k, v, beta, initial_state, scale)
    return output, final_state
