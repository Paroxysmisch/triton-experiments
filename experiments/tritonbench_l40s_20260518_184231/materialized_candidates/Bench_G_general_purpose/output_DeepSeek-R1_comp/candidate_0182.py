import torch
import triton
import triton.language as tl

@triton.jit
def fused_recurrent_hgrn_fwd_kernel(
    x_ptr, g_ptr, o_ptr,
    h0_ptr, ht_ptr,
    T, B, D,
    stride_x_batch, stride_x_time, stride_x_dim,
    stride_g_batch, stride_g_time, stride_g_dim,
    stride_o_batch, stride_o_time, stride_o_dim,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    BD: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_dim = tl.program_id(1)
    
    off_batch = pid_batch
    off_dim = pid_dim * BD
    dim_range = off_dim + tl.arange(0, BD)
    mask = dim_range < D
    
    if USE_INITIAL_STATE:
        h_prev = tl.load(h0_ptr + off_batch * D + dim_range, mask=mask, other=0.0)
    else:
        h_prev = tl.zeros([BD], dtype=tl.float32)
    
    for t in range(T):
        x_ptr_t = x_ptr + t * stride_x_time + off_batch * stride_x_batch + dim_range * stride_x_dim
        g_ptr_t = g_ptr + t * stride_g_time + off_batch * stride_g_batch + dim_range * stride_g_dim
        o_ptr_t = o_ptr + off_batch * stride_o_batch + t * stride_o_time + dim_range * stride_o_dim
        
        x_t = tl.load(x_ptr_t, mask=mask, other=0.0)
        g_t = tl.load(g_ptr_t, mask=mask, other=0.0)
        
        h_current = g_t * h_prev + x_t
        tl.store(o_ptr_t, h_current, mask=mask)
        h_prev = h_current
    
    if STORE_FINAL_STATE:
        tl.store(ht_ptr + off_batch * D + dim_range, h_prev, mask=mask)

@triton.jit
def fused_recurrent_hgrn_bwd_kernel(
    do_ptr, o_ptr, g_ptr,
    dx_ptr, dg_ptr,
    h0_ptr, dh0_ptr,
    T, B, D,
    stride_do_batch, stride_do_time, stride_do_dim,
    stride_o_batch, stride_o_time, stride_o_dim,
    stride_g_batch, stride_g_time, stride_g_dim,
    stride_dx_batch, stride_dx_time, stride_dx_dim,
    stride_dg_batch, stride_dg_time, stride_dg_dim,
    USE_INITIAL_STATE: tl.constexpr,
    BD: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_dim = tl.program_id(1)
    
    off_batch = pid_batch
    off_dim = pid_dim * BD
    dim_range = off_dim + tl.arange(0, BD)
    mask = dim_range < D
    
    dh = tl.zeros([BD], dtype=tl.float32)
    
    for t in range(T-1, -1, -1):
        do_ptr_t = do_ptr + off_batch * stride_do_batch + t * stride_do_time + dim_range * stride_do_dim
        g_ptr_t = g_ptr + off_batch * stride_g_batch + t * stride_g_time + dim_range * stride_g_dim
        
        do_t = tl.load(do_ptr_t, mask=mask, other=0.0)
        g_t = tl.load(g_ptr_t, mask=mask, other=0.0)
        dh += do_t
        
        dx_ptr_t = dx_ptr + off_batch * stride_dx_batch + t * stride_dx_time + dim_range * stride_dx_dim
        tl.store(dx_ptr_t, dh, mask=mask)
        
        if t == 0:
            if USE_INITIAL_STATE:
                h_prev = tl.load(h0_ptr + off_batch * D + dim_range, mask=mask, other=0.0)
            else:
                h_prev = tl.zeros([BD], dtype=tl.float32)
        else:
            o_ptr_prev = o_ptr + off_batch * stride_o_batch + (t-1) * stride_o_time + dim_range * stride_o_dim
            h_prev = tl.load(o_ptr_prev, mask=mask, other=0.0)
        
        dg_t = dh * h_prev
        dg_ptr_t = dg_ptr + off_batch * stride_dg_batch + t * stride_dg_time + dim_range * stride_dg_dim
        tl.store(dg_ptr_t, dg_t, mask=mask)
        
        dh = dh * g_t
    
    if USE_INITIAL_STATE:
        tl.store(dh0_ptr + off_batch * D + dim_range, dh, mask=mask)

class FusedRecurrentHGRNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, h0, use_initial_state, store_final_state):
        B, T, D = x.shape[1], x.shape[0], x.shape[2]
        BD = 64
        
        o = torch.empty_like(x)
        ht = torch.empty(B, D, device=x.device) if store_final_state else None
        
        grid = (B, triton.cdiv(D, BD))
        fused_recurrent_hgrn_fwd_kernel[grid](
            x, g, o,
            h0 if use_initial_state else None,
            ht if store_final_state else None,
            T, B, D,
            x.stride(1), x.stride(0), x.stride(2),
            g.stride(1), g.stride(0), g.stride(2),
            o.stride(1), o.stride(0), o.stride(2),
            USE_INITIAL_STATE=use_initial_state,
            STORE_FINAL_STATE=store_final_state,
            BD=BD,
        )
        
        saved_tensors = [o, g]
        if use_initial_state:
            saved_tensors.append(h0)
        ctx.save_for_backward(*saved_tensors)
        ctx.use_initial_state = use_initial_state
        ctx.store_final_state = store_final_state
        
        if store_final_state:
            return o, ht
        else:
            return o

    @staticmethod
    def backward(ctx, do, dht=None):
        saved_tensors = ctx.saved_tensors
        o, g = saved_tensors[:2]
        h0 = saved_tensors[2] if ctx.use_initial_state else None
        
        B, T, D = o.shape[1], o.shape[0], o.shape[2]
        BD = 64
        
        dx = torch.empty_like(do)
        dg = torch.empty_like(g)
        dh0 = torch.empty_like(h0) if ctx.use_initial_state else None
        
        grid = (B, triton.cdiv(D, BD))
        fused_recurrent_hgrn_bwd_kernel[grid](
            do, o, g,
            dx, dg,
            h0, dh0 if ctx.use_initial_state else None,
            T, B, D,
            do.stride(1), do.stride(0), do.stride(2),
            o.stride(1), o.stride(0), o.stride(2),
            g.stride(1), g.stride(0), g.stride(2),
            dx.stride(1), dx.stride(0), dx.stride(2),
            dg.stride(1), dg.stride(0), dg.stride(2),
            USE_INITIAL_STATE=ctx.use_initial_state,
            BD=BD,
        )
        
        return dx, dg, dh0 if ctx.use_initial_state else None, None, None

def fused_recurrent_hgrn(x, g, h0=None, use_initial_state=False, store_final_state=False):
    if h0 is None:
        use_initial_state = False
    x = x.contiguous()
    g = g.contiguous()
    if use_initial_state:
        h0 = h0.contiguous()
    o = FusedRecurrentHGRNFunction.apply(x, g, h0, use_initial_state, store_final_state)
    if store_final_state:
        return o[0], o[1]
    else:
        return o
