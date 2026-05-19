import triton
import triton.language as tl

@triton.jit
def _swiglu_bwd_kernel(
    X, Y, DX, DY, DOUT, OUT, 
    stride_xm, stride_ym, stride_dxm, stride_dym, stride_doutm, stride_outm,
    stride_xn, stride_yn, stride_dxn, stride_dyn, stride_doutn, stride_outn,
    M, N, BLOCK_N: tl.constexpr, RECOMPUTE_OUTPUT: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_N

    # Initialize offsets
    offsets_m = tl.arange(0, BLOCK_N) + block_start
    offsets_n = tl.arange(0, BLOCK_N)

    # Create pointers for X, Y, DX, DY, DOUT, and OUT
    x_ptrs = X + offsets_m[:, None] * stride_xm + offsets_n[None, :] * stride_xn
    y_ptrs = Y + offsets_m[:, None] * stride_ym + offsets_n[None, :] * stride_yn
    dx_ptrs = DX + offsets_m[:, None] * stride_dxm + offsets_n[None, :] * stride_dxn
    dy_ptrs = DY + offsets_m[:, None] * stride_dym + offsets_n[None, :] * stride_dyn
    dout_ptrs = DOUT + offsets_m[:, None] * stride_doutm + offsets_n[None, :] * stride_doutn
    out_ptrs = OUT + offsets_m[:, None] * stride_outm + offsets_n[None, :] * stride_outn

    # Load data with boundary checks
    x = tl.load(x_ptrs, mask=offsets_m[:, None] < M, other=0.0)
    y = tl.load(y_ptrs, mask=offsets_m[:, None] < M, other=0.0)
    dout = tl.load(dout_ptrs, mask=offsets_m[:, None] < M, other=0.0)

    # Compute sigmoid and Swish derivative
    sigmoid_x = 1 / (1 + tl.exp(-x))
    swish_derivative = sigmoid_x * (1 + x * (1 - sigmoid_x))

    # Compute gradients
    dx = swish_derivative * y * dout
    dy = sigmoid_x * x * dout

    # Optionally recompute and store the output
    if RECOMPUTE_OUTPUT:
        out = sigmoid_x * x * y
        tl.store(out_ptrs, out, mask=offsets_m[:, None] < M)

    # Store gradients
    tl.store(dx_ptrs, dx, mask=offsets_m[:, None] < M)
    tl.store(dy_ptrs, dy, mask=offsets_m[:, None] < M)

import torch
from torch.autograd import Function

class SwigluBwd(Function):
    @staticmethod
    def forward(ctx, x, y, dout, recompute_output=False):
        x = x.contiguous()
        y = y.contiguous()
        dout = dout.contiguous()

        B, M, N = x.shape
        x = x.view(-1, N)
        y = y.view(-1, N)
        dout = dout.view(-1, N)

        DX = torch.zeros_like(x)
        DY = torch.zeros_like(y)
        OUT = torch.zeros_like(x) if recompute_output else None

        grid = (M,)

        _swiglu_bwd_kernel[grid](
            x, y, DX, DY, dout, OUT,
            x.stride(0), y.stride(0), DX.stride(0), DY.stride(0), dout.stride(0), OUT.stride(0) if OUT is not None else 0,
            x.stride(1), y.stride(1), DX.stride(1), DY.stride(1), dout.stride(1), OUT.stride(1) if OUT is not None else 0,
            M, N, BLOCK_N=128, RECOMPUTE_OUTPUT=recompute_output
        )

        DX = DX.view(B, M, N)
        DY = DY.view(B, M, N)
        if recompute_output:
            OUT = OUT.view(B, M, N)

        ctx.save_for_backward(x, y, dout, OUT if recompute_output else None)
        return DX, DY, OUT if recompute_output else None

    @staticmethod
    def backward(ctx, grad_DX, grad_DY, grad_OUT=None):
        x, y, dout, out = ctx.saved_tensors
        # Implement the backward pass if needed
        return None, None, None, None

def swiglu_bwd(x, y, dout, recompute_output=False):
    return SwigluBwd.apply(x, y, dout, recompute_output)
