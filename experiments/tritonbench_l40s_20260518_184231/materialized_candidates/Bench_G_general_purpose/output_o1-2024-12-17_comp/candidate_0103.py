import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024
num_warps = 4

@triton.jit
def _rms_layernorm_forward(
    X_PTR, W_PTR, OUT_PTR,
    mean_INV_PTR,  # store inverse sqrt of RMS for backward usage
    N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_PTR + offs, mask=mask, other=0.0)

    # compute row-wise rms
    sq = x * x
    sum_sq = tl.sum(sq, axis=0)
    rmean = tl.sqrt(sum_sq / N)
    inv_rmean = 1.0 / (rmean + 1e-5)

    # store inv_rmean for backward
    if tl.first(offs) < N:
        tl.store(mean_INV_PTR + pid, inv_rmean)

    # normalize and scale
    w = tl.load(W_PTR + offs, mask=mask, other=0.0)
    y = x * inv_rmean * w
    tl.store(OUT_PTR + offs, y, mask=mask)

@triton.jit
def _rms_layernorm_backward(
    X_PTR, W_PTR, dY_PTR,
    mean_INV_PTR, dX_PTR, dW_PTR,
    N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(X_PTR + offs, mask=mask, other=0.0)
    w = tl.load(W_PTR + offs, mask=mask, other=0.0)
    dy = tl.load(dY_PTR + offs, mask=mask, other=0.0)

    inv_rmean = tl.load(mean_INV_PTR + pid)
    dx_norm = dy * w * inv_rmean

    # sum for partial derivative wrt RMS
    sum_dx_norm_x = tl.sum(dx_norm * x, axis=0)
    # compute derivative wrt x
    d_x = dx_norm - x * (sum_dx_norm_x / (N * (inv_rmean * inv_rmean)))
    tl.store(dX_PTR + offs, d_x, mask=mask)

    # compute derivative wrt w
    d_w = dy * x * inv_rmean
    tl.store(dW_PTR + offs, d_w, mask=mask)

@triton.jit
def _gemma_rms_layernorm_forward(
    X_PTR, W_PTR, OUT_PTR,
    mean_INV_PTR,
    N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_PTR + offs, mask=mask, other=0.0)

    # compute row-wise rms
    sq = x * x
    sum_sq = tl.sum(sq, axis=0)
    rmean = tl.sqrt(sum_sq / N)
    inv_rmean = 1.0 / (rmean + 1e-5)

    # store inv_rmean for backward
    if tl.first(offs) < N:
        tl.store(mean_INV_PTR + pid, inv_rmean)

    # normalize and scale with W + 1.0
    w = tl.load(W_PTR + offs, mask=mask, other=0.0) + 1.0
    y = x * inv_rmean * w
    tl.store(OUT_PTR + offs, y, mask=mask)

class Fast_RMS_LayernormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, gemma=False):
        x_ = x.contiguous()
        w_ = w.contiguous()
        B = x_.shape[0]
        size = x_.shape[1]
        out = torch.empty_like(x_)
        mean_inv = torch.empty(B, device=x_.device, dtype=x_.dtype)

        grid = lambda META: (B,)
        if not gemma:
            _rms_layernorm_forward[grid](
                x_, w_, out, mean_inv,
                size,
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            _gemma_rms_layernorm_forward[grid](
                x_, w_, out, mean_inv,
                size,
                BLOCK_SIZE=BLOCK_SIZE
            )

        ctx.save_for_backward(x_, w_, mean_inv)
        ctx.gemma = gemma
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_, w_, mean_inv = ctx.saved_tensors
        B, size = x_.shape
        dx = torch.empty_like(x_)
        dw = torch.empty_like(x_)
        grid = lambda META: (B,)
        _rms_layernorm_backward[grid](
            x_, w_, grad_output, mean_inv,
            dx, dw,
            size,
            BLOCK_SIZE=BLOCK_SIZE
        )
        # sum dw over each row
        grad_w = dw.sum(dim=0)
        return dx, grad_w, None

def calculate_settings(x_shape):
    # For demonstration, we keep them constant. Could be tuned for better perf.
    return BLOCK_SIZE, num_warps

def fast_rms_layernorm(x, w, gemma=False):
    return Fast_RMS_LayernormFunction.apply(x, w, gemma)
