import torch
import triton
import triton.language as tl
from torch.autograd.function import Function

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
    # Triton kernel for forward pass of fused recurrent HGRN
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_x = x + i_bh * T * D + o_d
    p_g = g + i_bh * T * D + o_d
    p_o = o + i_bh * T * D + o_d

    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * D + o_d
        b_h += tl.load(p_h0, mask=mask, other=0).to(tl.float32)
    for _ in range(0, T):
        b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_h = b_g * b_h + b_x
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask)

        p_x += D
        p_g += D
        p_o += D

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * D + o_d
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask)

@triton.jit
def fused_recurrent_hgrn_bwd_kernel(
    x,
    g,
    o,
    do,
    dx,
    dg,
    h0,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
):
    # Triton kernel for backward pass of fused recurrent HGRN
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    b_dh = tl.zeros([BD], dtype=tl.float32)

    p_x = x + (i_bh * T + T - 1) * D + o_d
    p_g = g + (i_bh * T + T - 2) * D + o_d
    p_o = o + (i_bh * T + T - 2) * D + o_d
    p_do = do + (i_bh * T + T - 1) * D + o_d
    p_dx = dx + (i_bh * T + T - 1) * D + o_d
    p_dg = dg + (i_bh * T + T - 1) * D + o_d

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * D + o_d
        b_h = tl.load(p_h0, mask=mask, other=0).to(tl.float32)
    else:
        b_h = tl.zeros([BD], dtype=tl.float32)

    for i in range(T - 1, -1, -1):
        b_dh = b_dh + tl.load(p_do, mask=mask, other=0).to(tl.float32)
        b_dx = b_dh
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)

        b_dg = b_dh * tl.load(p_g, mask=mask, other=0).to(tl.float32)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=mask)

        b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
        b_o = tl.load(p_o, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_h = b_g * b_h + b_x
        b_dh = b_dh * b_g

        p_x -= D
        p_g -= D
        p_o -= D
        p_do -= D
        p_dx -= D
        p_dg -= D

class FusedRecurrentHGRNFunction(Function):
    @staticmethod
    def forward(ctx, x, g, initial_state=None, output_final_state=False):
        B, H, T, D = x.shape
        BD = 16
        NT = triton.cdiv(D, BD)

        o = torch.empty_like(x)
        if output_final_state:
            ht = torch.empty(B, H, D, dtype=x.dtype, device=x.device)
        else:
            ht = None

        kwargs = [
            x,
            g,
            o,
            initial_state,
            ht,
            T,
            D,
            BD,
            initial_state is not None,
            output_final_state,
        ]

        fused_recurrent_hgrn_fwd_kernel[NT, B * H](*kwargs)

        ctx.save_for_backward(x, g, initial_state)
        ctx.BD = BD
        ctx.NT = NT
        return o, ht

    @staticmethod
    def backward(ctx, do, dht=None):
        x, g, initial_state = ctx.saved_tensors
        B, H, T, D = x.shape

        BD = 16
        NT = triton.cdiv(D, BD)

        dx = torch.empty_like(x)
        dg = torch.empty_like(g)
        fused_recurrent_hgrn_bwd_kernel[NT, B * H](
            x,
            g,
            o,
            do,
            dx,
            dg,
            initial_state,
            T,
            D,
            BD,
            initial_state is not None,
        )
        return dx, dg, None, None

def fused_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, _ = FusedRecurrentHGRNFunction.apply(
        x, g, initial_state, output_final_state
    )
    return o
