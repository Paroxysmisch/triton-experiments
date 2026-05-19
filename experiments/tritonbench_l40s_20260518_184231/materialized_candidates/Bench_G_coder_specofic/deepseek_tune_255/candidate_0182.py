import torch
import triton
import triton.language as tl

@triton.jit
def fused_recurrent_hgrn_fwd_kernel(
    x,
    g,
    o,
    h0,
    ht,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_x = x + i_bh * T * D + o_d
    p_g = g + i_bh * T * D + o_d
    p_o = o + i_bh * T * D + o_d
    p_ht = ht + i_bh * D + o_d

    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * D + o_d
        b_h += tl.load(p_h0, mask=mask, other=0).to(tl.float32)

    for i in range(0, T):
        b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_h = b_h * b_g + b_x
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask)

        p_x += D
        p_g += D
        p_o += D

    if STORE_FINAL_STATE:
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask)

@triton.jit
def fused_recurrent_hgrn_bwd_kernel(
    g,
    do,
    dh,
    dg,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
):
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_do = do + i_bh * T * D + o_d
    p_dh = dh + i_bh * T * D + o_d
    p_dg = dg + i_bh * T * D + o_d

    if USE_INITIAL_STATE:
        p_dh += (T - 1) * D

    b_dh = tl.zeros([BD], dtype=tl.float32)
    for i in range(T - 1, -1, -1):
        b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)
        b_dh += b_do
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), mask=mask)

        b_dh * tl.load(g + i_bh * T * D + o_d + (i * D), mask=mask, other=0).to(
            tl.float32
        )
        tl.store(p_dg, b_dh.to(p_dg.dtype.element_ty), mask=mask)

        p_do -= D
        p_dh -= D
        p_dg -= D

class FusedRecurrentHGRNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, h0, initial_state_flag):
        B, H, T, D = x.shape
        BD = triton.next_power_of_2(D)
        num_warps = 4
        if BD > 2047:
            num_warps = 8
        if BD > 4095:
            num_warps = 16

        o = torch.empty((B, H, T, D), device=x.device, dtype=x.dtype)
        if initial_state_flag:
            h0 = h0.unsqueeze(1).expand(-1, H, -1, -1).reshape(-1, H * D)
            ht = torch.empty((B, H, D), device=x.device, dtype=x.dtype)
            fused_recurrent_hgrn_fwd_kernel[
                (D, H)
            ](
                x,
                g,
                o,
                h0,
                ht,
                T,
                D,
                BD=BD,
                USE_INITIAL_STATE=True,
                STORE_FINAL_STATE=True,
                num_warps=num_warps,
            )
            ht = ht.unsqueeze(2).expand(-1, -1, T, -1).reshape(B, H, T, D)
            ctx.save_for_backward(o, g, ht)
            return o, ht
        else:
            fused_recurrent_hgrn_fwd_kernel[
                (D, H)
            ](
                x,
                g,
                o,
                h0,
                ht,
                T,
                D,
                BD=BD,
                USE_INITIAL_STATE=False,
                STORE_FINAL_STATE=False,
                num_warps=num_warps,
            )
            ctx.save_for_backward(o, g)
            return o

    @staticmethod
    def backward(ctx, do=None, dh=None, dht=None, initial_state_flag=None):
        B, H, T, D = do.shape
        BD = triton.next_power_of_2(D)
        num_warps = 4
        if BD > 2047:
            num_warps = 8
        if BD > 4095:
            num_warps = 16

        if initial_state_flag:
            (o, g, ht) = ctx.saved_tensors
            dx = torch.empty((B, H, T, D), device=do.device, dtype=do.dtype)
            dg = torch.empty((B, H, T, D), device=do.device, dtype=do.dtype)
            dh = torch.empty((B, H, T, D), device=do.device, dtype=do.dtype)
            fused_recurrent_hgrn_bwd_kernel[
                (D, H)
            ](
                g,
                do,
                dh,
                dg,
                T,
                D,
                BD=BD,
                USE_INITIAL_STATE=True,
                num_warps=num_warps,
            )
            fused_recurrent_hgrn_bwd_kernel[
                (D, H)
            ](
                g,
                dht,
                dh,
                dg,
                T,
                D,
                BD=BD,
                USE_INITIAL_STATE=True,
                num_warps=num_warps,
            )
            ctx.save_for_backward(dh)
            return dx, dg, dh, None
        else:
            (o, g) = ctx.saved_tensors
            dx = torch.empty((B, H, T, D), device=do.device, dtype=do.dtype)
            dg = torch.empty
