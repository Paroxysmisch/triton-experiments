import torch
import triton
import triton.language as tl

@triton.jit
def fused_recurrent_hgrn_fwd_kernel(
    x_ptr, g_ptr, o_ptr, h0_ptr, ht_ptr,
    T, D, BD, USE_INITIAL_STATE, STORE_FINAL_STATE,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D
    h = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    if USE_INITIAL_STATE != 0:
        h = tl.load(h0_ptr + offs, mask=mask, other=h)

    for t in range(T):
        x_offset = t * D + offs
        g_offset = t * D + offs
        x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
        g_val = tl.load(g_ptr + g_offset, mask=mask, other=0.0)
        h = g_val * h + x_val
        out_offset = t * D + offs
        tl.store(o_ptr + out_offset, h, mask=mask)

    if STORE_FINAL_STATE != 0:
        tl.store(ht_ptr + offs, h, mask=mask)

@triton.jit
def fused_recurrent_hgrn_bwd_kernel(
    x_ptr, g_ptr, o_ptr, do_ptr, dh_ptr,
    dx_ptr, dg_ptr,
    T, D, BD,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D
    dh = tl.load(dh_ptr + offs, mask=mask, other=0.0)

    for t in range(T - 1, -1, -1):
        do_offset = t * D + offs
        x_offset = t * D + offs
        g_offset = t * D + offs
        o_offset = t * D + offs

        do_val = tl.load(do_ptr + do_offset, mask=mask, other=0.0)
        x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
        g_val = tl.load(g_ptr + g_offset, mask=mask, other=0.0)
        o_val = tl.load(o_ptr + o_offset, mask=mask, other=0.0)

        dh = dh + do_val
        dx_val = dh
        dg_val = dh * o_val
        dh = dh * g_val

        tl.store(dx_ptr + x_offset, dx_val, mask=mask)
        tl.store(dg_ptr + g_offset, dg_val, mask=mask)

class FusedRecurrentHGRNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, h0=None, use_initial_state=False, store_final_state=False):
        T, D = x.shape
        x_out = torch.empty_like(x)
        if h0 is None:
            h0 = torch.zeros(D, device=x.device, dtype=x.dtype)
        ht_out = torch.zeros_like(h0) if store_final_state else None
        grid = lambda meta: ( (D + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )

        fused_recurrent_hgrn_fwd_kernel[grid](
            x, g, x_out, h0, ht_out if ht_out is not None else h0,
            T, D, D,
            int(use_initial_state),
            int(store_final_state),
            BLOCK_SIZE=1024
        )

        ctx.save_for_backward(x, g, x_out, h0, torch.tensor([use_initial_state], device=x.device),
                             torch.tensor([store_final_state], device=x.device))
        return (x_out, ht_out) if store_final_state else (x_out,)

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, g, o, h0, use_initial_state, store_final_state = ctx.saved_tensors
        T, D = x.shape
        dx = torch.empty_like(x)
        dg = torch.empty_like(g)
        do = grad_outputs[0]
        if store_final_state.item() != 0.0:
            dh = grad_outputs[1] if len(grad_outputs) > 1 else torch.zeros(D, device=x.device, dtype=x.dtype)
        else:
            dh = torch.zeros(D, device=x.device, dtype=x.dtype)
        grid = lambda meta: ( (D + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )

        fused_recurrent_hgrn_bwd_kernel[grid](
            x, g, o, do, dh,
            dx, dg,
            T, D, D,
            BLOCK_SIZE=1024
        )

        dh0 = None
        if use_initial_state.item() != 0.0:
            dh0 = dh

        return dx, dg, dh0, None, None

def fused_recurrent_hgrn(x, g, h0=None, store_final_state=False):
    if h0 is not None:
        x_out, ht_out = FusedRecurrentHGRNFunction.apply(x.detach(), g.detach(), h0.detach(), True, store_final_state)
        return (x_out, ht_out)
    else:
        out = FusedRecurrentHGRNFunction.apply(x.detach(), g.detach(), None, False, store_final_state)
        return out if len(out) > 1 else out[0]
