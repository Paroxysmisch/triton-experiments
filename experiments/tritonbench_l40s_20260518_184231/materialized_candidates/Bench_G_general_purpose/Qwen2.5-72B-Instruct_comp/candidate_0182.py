import triton
import triton.language as tl

@triton.jit
def fused_recurrent_hgrn_fwd_kernel(
    x_ptr, g_ptr, o_ptr, h0_ptr, ht_ptr,
    T, D, BD,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BD
    for t in range(T):
        for i in range(block_start, block_start + BD):
            if i < D:
                if t == 0 and USE_INITIAL_STATE:
                    b_h = tl.load(h0_ptr + i)
                else:
                    b_h = tl.load(o_ptr + (t - 1) * D + i)
                b_x = tl.load(x_ptr + t * D + i)
                b_g = tl.load(g_ptr + t * D + i)
                b_h = b_g * b_h + b_x
                tl.store(o_ptr + t * D + i, b_h)
        if t == T - 1 and STORE_FINAL_STATE:
            for i in range(block_start, block_start + BD):
                if i < D:
                    b_h = tl.load(o_ptr + (T - 1) * D + i)
                    tl.store(ht_ptr + i, b_h)

@triton.jit
def fused_recurrent_hgrn_bwd_kernel(
    x_ptr, g_ptr, o_ptr, dx_ptr, dg_ptr, do_ptr,
    T, D, BD
):
    pid = tl.program_id(axis=0)
    block_start = pid * BD
    b_dh = tl.zeros((BD,), dtype=tl.float32)
    for t in range(T - 1, -1, -1):
        for i in range(block_start, block_start + BD):
            if i < D:
                b_do = tl.load(do_ptr + t * D + i)
                b_dh = b_dh + b_do
                b_o = tl.load(o_ptr + t * D + i)
                b_dx = b_dh
                b_dg = b_dh * b_o
                b_dh = b_dh * tl.load(g_ptr + t * D + i)
                tl.store(dx_ptr + t * D + i, b_dx)
                tl.store(dg_ptr + t * D + i, b_dg)

import torch
from torch.autograd import Function

class FusedRecurrentHGRNFunction(Function):
    @staticmethod
    def forward(ctx, x, g, h0=None, ht=None, T=None, D=None, BD=None, USE_INITIAL_STATE=False, STORE_FINAL_STATE=False):
        assert x.is_cuda and g.is_cuda
        assert x.shape == (T, D) and g.shape == (T, D)
        o = torch.empty_like(x)
        if h0 is not None:
            assert h0.shape == (D,)
            h0 = h0.contiguous()
        if ht is not None:
            assert ht.shape == (D,)
            ht = ht.contiguous()

        grid = (D // BD, 1)
        fused_recurrent_hgrn_fwd_kernel[grid](
            x, g, o, h0, ht,
            T, D, BD,
            USE_INITIAL_STATE, STORE_FINAL_STATE
        )

        ctx.save_for_backward(x, g, o)
        ctx.T, ctx.D, ctx.BD = T, D, BD
        ctx.USE_INITIAL_STATE = USE_INITIAL_STATE
        ctx.STORE_FINAL_STATE = STORE_FINAL_STATE

        return o, ht if STORE_FINAL_STATE else o

    @staticmethod
    def backward(ctx, do, dht=None):
        x, g, o = ctx.saved_tensors
        T, D, BD = ctx.T, ctx.D, ctx.BD
        dx = torch.zeros_like(x)
        dg = torch.zeros_like(g)
        if dht is not None:
            assert dht.shape == (D,)
            dht = dht.contiguous()

        grid = (D // BD, 1)
        fused_recurrent_hgrn_bwd_kernel[grid](
            x, g, o, dx, dg, do,
            T, D, BD
        )

        if ctx.STORE_FINAL_STATE:
            return dx, dg, None, dht, None, None, None, None, None
        else:
            return dx, dg, None, None, None, None, None, None, None

def fused_recurrent_hgrn(x, g, h0=None, ht=None, T=None, D=None, BD=None, USE_INITIAL_STATE=False, STORE_FINAL_STATE=False):
    if h0 is not None:
        h0 = h0.detach()
    if ht is not None:
        ht = ht.detach()
    return FusedRecurrentHGRNFunction.apply(x, g, h0, ht, T, D, BD, USE_INITIAL_STATE, STORE_FINAL_STATE)

import torch

# Example usage
T, D, BD = 10, 128, 32
x = torch.randn(T, D, device='cuda')
g = torch.randn(T, D, device='cuda')
h0 = torch.randn(D, device='cuda') if USE_INITIAL_STATE else None
ht = torch.empty(D, device='cuda') if STORE_FINAL_STATE else None

output, final_state = fused_recurrent_hgrn(x, g, h0, ht, T, D, BD, USE_INITIAL_STATE=True, STORE_FINAL_STATE=True)

# Forward pass
print(output)

# Backward pass
output.sum().backward()
print(x.grad)
print(g.grad)
