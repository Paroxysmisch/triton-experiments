import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def fused_recurrent_hgrn_fwd_kernel(
    x_ptr, g_ptr, o_ptr, h0_ptr, ht_ptr,
    stride_xb, stride_xh, stride_xt, stride_xd,
    stride_gb, stride_gh, stride_gt, stride_gd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_h0b, stride_h0h, stride_h0d,
    stride_htb, stride_hth, stride_htd,
    T: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr
):
    pid_d = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    d_start = pid_d * BD
    d_offsets = d_start + tl.arange(0, BD)
    d_mask = d_offsets < D

    b_id = pid_bh // stride_xh
    h_id = pid_bh % stride_xh

    x_block_ptr = x_ptr + b_id * stride_xb + h_id * stride_xh + d_start
    g_block_ptr = g_ptr + b_id * stride_gb + h_id * stride_gh + d_start
    o_block_ptr = o_ptr + b_id * stride_ob + h_id * stride_oh + d_start

    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        h0_block_ptr = h0_ptr + b_id * stride_h0b + h_id * stride_h0h + d_start
        b_h = tl.load(h0_block_ptr, mask=d_mask, other=0.0)

    for t in range(T):
        b_x = tl.load(x_block_ptr + t * stride_xt, mask=d_mask, other=0.0)
        b_g = tl.load(g_block_ptr + t * stride_gt, mask=d_mask, other=0.0)
        
        b_h = tl.exp(b_g) * b_h + b_x
        tl.store(o_block_ptr + t * stride_ot, b_h, mask=d_mask)

    if STORE_FINAL_STATE:
        ht_block_ptr = ht_ptr + b_id * stride_htb + h_id * stride_hth + d_start
        tl.store(ht_block_ptr, b_h, mask=d_mask)

@triton.jit
def fused_recurrent_hgrn_bwd_kernel(
    g_ptr, o_ptr, dx_ptr, dg_ptr, do_ptr, h0_ptr,
    stride_gb, stride_gh, stride_gt, stride_gd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_dxb, stride_dxh, stride_dxt, stride_dxd,
    stride_dgb, stride_dgh, stride_dgt, stride_dgd,
    stride_dob, stride_doh, stride_dot, stride_dod,
    stride_h0b, stride_h0h, stride_h0d,
    T: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr
):
    pid_d = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    d_start = pid_d * BD
    d_offsets = d_start + tl.arange(0, BD)
    d_mask = d_offsets < D

    b_id = pid_bh // stride_gh
    h_id = pid_bh % stride_gh

    g_block_ptr = g_ptr + b_id * stride_gb + h_id * stride_gh + d_start
    o_block_ptr = o_ptr + b_id * stride_ob + h_id * stride_oh + d_start
    dx_block_ptr = dx_ptr + b_id * stride_dxb + h_id * stride_dxh + d_start
    dg_block_ptr = dg_ptr + b_id * stride_dgb + h_id * stride_dgh + d_start
    do_block_ptr = do_ptr + b_id * stride_dob + h_id * stride_doh + d_start

    b_dh = tl.zeros([BD], dtype=tl.float32)
    
    for t in range(T-1, -1, -1):
        b_g = tl.load(g_block_ptr + t * stride_gt, mask=d_mask, other=0.0)
        b_do = tl.load(do_block_ptr + t * stride_dot, mask=d_mask, other=0.0)
        
        if t > 0:
            b_o = tl.load(o_block_ptr + (t-1) * stride_ot, mask=d_mask, other=0.0)
        elif USE_INITIAL_STATE:
            h0_block_ptr = h0_ptr + b_id * stride_h0b + h_id * stride_h0h + d_start
            b_o = tl.load(h0_block_ptr, mask=d_mask, other=0.0)
        else:
            b_o = tl.zeros([BD], dtype=tl.float32)

        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dg = b_dh * b_o
        b_dh = b_dh * tl.exp(b_g)

        tl.store(dx_block_ptr + t * stride_dxt, b_dx, mask=d_mask)
        tl.store(dg_block_ptr + t * stride_dgt, b_dg, mask=d_mask)

class FusedRecurrentHGRNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, initial_state=None, output_final_state=False):
        B, H, T, D = x.shape
        o = torch.empty_like(x)
        final_state = torch.empty(B, H, D, device=x.device, dtype=x.dtype) if output_final_state else None
        
        BLOCK_SIZE = 32
        grid = (triton.cdiv(D, BLOCK_SIZE), B * H)
        
        fused_recurrent_hgrn_fwd_kernel[grid](
            x, g, o, initial_state, final_state,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            initial_state.stride(0) if initial_state is not None else 0,
            initial_state.stride(1) if initial_state is not None else 0,
            initial_state.stride(2) if initial_state is not None else 0,
            final_state.stride(0) if final_state is not None else 0,
            final_state.stride(1) if final_state is not None else 0,
            final_state.stride(2) if final_state is not None else 0,
            T=T, D=D, BD=BLOCK_SIZE,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
        )
        
        ctx.save_for_backward(x, g, o, initial_state)
        return o, final_state

    @staticmethod
    def backward(ctx, do, dht=None):
        x, g, o, initial_state = ctx.saved_tensors
        B, H, T, D = do.shape
        
        dx = torch.empty_like(x)
        dg = torch.empty_like(g)
        
        BLOCK_SIZE = 32
        grid = (triton.cdiv(D, BLOCK_SIZE), B * H)
        
        fused_recurrent_hgrn_bwd_kernel[grid](
            g, o, dx, dg, do, initial_state,
            g.stride(0), g.stride(1), g.stride(2), g.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            dg.stride(0), dg.stride(1), dg.stride(2), dg.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            initial_state.stride(0) if initial_state is not None else 0,
            initial_state.stride(1) if initial_state is not None else 0,
            initial_state.stride(2) if initial_state is not None else 0,
            T=T, D=D, BD=BLOCK_SIZE,
            USE_INITIAL_STATE=initial_state is not None,
        )
        
        return dx, dg, None, None

def fused_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = FusedRecurrentHGRNFunction.apply(x, g, initial_state, output_final_state)
    return o, final_state
