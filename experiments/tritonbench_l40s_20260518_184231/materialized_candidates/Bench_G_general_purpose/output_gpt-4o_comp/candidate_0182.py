import triton
import triton.language as tl

@triton.jit
def fused_recurrent_hgrn_fwd_kernel(x_ptr, g_ptr, o_ptr, h0_ptr, ht_ptr,
                                    T, D, BD, USE_INITIAL_STATE, STORE_FINAL_STATE,
                                    stride_x, stride_g, stride_o, stride_h0, stride_ht):
    pid = tl.program_id(axis=0)
    
    # Offsets for this program instance
    offset = pid * BD
    b_x = tl.load(x_ptr + offset, mask=offset < D)
    b_g = tl.load(g_ptr + offset, mask=offset < D)
    
    # Initialize hidden state
    if USE_INITIAL_STATE:
        b_h = tl.load(h0_ptr + offset, mask=offset < D)
    else:
        b_h = tl.zeros([BD], dtype=tl.float32)
    
    for t in range(T):
        b_h = b_g * b_h + b_x
        tl.store(o_ptr + offset + t * stride_o, b_h, mask=offset < D)
        # Update b_x and b_g for the next timestep
        b_x = tl.load(x_ptr + offset + (t+1) * stride_x, mask=offset < D)
        b_g = tl.load(g_ptr + offset + (t+1) * stride_g, mask=offset < D)
    
    if STORE_FINAL_STATE:
        tl.store(ht_ptr + offset, b_h, mask=offset < D)

@triton.jit
def fused_recurrent_hgrn_bwd_kernel(x_ptr, g_ptr, o_ptr, do_ptr, dx_ptr, dg_ptr, ht_ptr,
                                    T, D, BD, stride_x, stride_g, stride_o, stride_do, stride_dx, stride_dg):
    pid = tl.program_id(axis=0)
    
    # Offsets for this program instance
    offset = pid * BD
    b_dh = tl.zeros([BD], dtype=tl.float32)
    
    for t in range(T-1, -1, -1):
        b_do = tl.load(do_ptr + offset + t * stride_do, mask=offset < D)
        b_o = tl.load(o_ptr + offset + t * stride_o, mask=offset < D)
        b_g = tl.load(g_ptr + offset + t * stride_g, mask=offset < D)
        
        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dg = b_dh * b_o
        b_dh = b_dh * b_g
        
        tl.store(dx_ptr + offset + t * stride_dx, b_dx, mask=offset < D)
        tl.store(dg_ptr + offset + t * stride_dg, b_dg, mask=offset < D)

import torch
from torch.autograd import Function

class FusedRecurrentHGRNFunction(Function):
    @staticmethod
    def forward(ctx, x, g, h0=None, store_final_state=False):
        T, D = x.shape
        BD = 32  # Block dimension
        USE_INITIAL_STATE = h0 is not None
        STORE_FINAL_STATE = store_final_state
        
        o = torch.empty_like(x)
        ht = torch.empty_like(h0) if STORE_FINAL_STATE else None
        
        # Launch the forward kernel
        grid = lambda meta: (triton.cdiv(D, BD),)
        fused_recurrent_hgrn_fwd_kernel[grid](
            x, g, o, h0, ht, T, D, BD, USE_INITIAL_STATE, STORE_FINAL_STATE,
            x.stride(0), g.stride(0), o.stride(0), h0.stride(0) if h0 is not None else 0, ht.stride(0) if ht is not None else 0
        )
        
        ctx.save_for_backward(x, g, o, ht)
        ctx.T, ctx.D, ctx.BD = T, D, BD
        
        return (o, ht) if STORE_FINAL_STATE else o

    @staticmethod
    def backward(ctx, do, dht=None):
        x, g, o, ht = ctx.saved_tensors
        T, D, BD = ctx.T, ctx.D, ctx.BD
        
        dx = torch.empty_like(x)
        dg = torch.empty_like(g)
        
        # Launch the backward kernel
        grid = lambda meta: (triton.cdiv(D, BD),)
        fused_recurrent_hgrn_bwd_kernel[grid](
            x, g, o, do, dx, dg, ht,
            T, D, BD,
            x.stride(0), g.stride(0), o.stride(0), do.stride(0), dx.stride(0), dg.stride(0)
        )
        
        return dx, dg, None, None

def fused_recurrent_hgrn(x, g, h0=None, store_final_state=False):
    return FusedRecurrentHGRNFunction.apply(x, g, h0, store_final_state)
