,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr
):
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_g = g + (i_bh * T + T - 1) * D + o_d
    p_o = o + (i_bh * T + T - 2) * D + o_d
    p_dx = dx + (i_bh * T + T - 1) * D + o_d
    p_dg = dg + (i_bh * T + T - 1) * D + o_d
    p_do = do + (i_bh * T + T - 1) * D + o_d

    b_dh = tl.zeros([BD], dtype=tl.float32)
    for i in range(T - 1, -1, -1):
        b_dh += tl.load(p_do, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_o = tl.load(p_o, mask=mask, other=0).to(tl.float32)
        b_dx = b_dh
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)

        b_dh = b_dh * tl.exp(b_g)
        b_dg = b_dh * b_o
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=mask)

        p_g -= D
        p_o -= D
        p_dx -= D
        p_dg -= D
        p_do -= D

class FusedRecurrentHGRNFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, g, initial_state=None, output_final_state=False):
        B, H, T, D = x.shape

        BD = 128
        num_warps = 4
        if D <= 512:
            num_warps = 8
        if D <= 256:
            num_warps = 16
        if D <= 128:
            num_warps = 32

        final_state = None
        if output_final_state:
            final_state = x.new_empty(B, H, D)

        o = torch.empty_like(x)
        grid = (triton.cdiv(D, BD), B * H)
        fused_recurrent_hgrn_fwd_kernel[grid](
            x,
            g,
            o,
            initial_state,
            final_state,
            T,
            D,
            BD,
            initial_state is not None,
            output_final_state
        )
        ctx.save_for_backward(g, o, initial_state)
        return o, final_state

    @staticmethod
    def backward(ctx, do, dht=None):
        g, o, initial_state = ctx.saved_tensors
        B, H, T, D = do.shape

        BD = 128
        num_warps = 4
        if D <= 512:
            num_warps = 8
        if D <= 256:
            num_warps = 16
        if D <= 128:
            num_warps = 32

        dx = torch.empty_like(o)
        dg = torch.empty_like(g)
        grid = (triton.cdiv(D, BD), B * H)
        fused_recurrent_hgrn_bwd_kernel[grid](
            g,
            o,
            dx,
            dg,
            do,
            initial_state,
            T,
            D,
            BD,
            initial_state is not None
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
    return o, final_state if output_final_state else o
