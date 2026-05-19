import torch
import triton
import triton.language as tl

def calculate_settings(n):
    MAX_FUSED_SIZE = 65536 // 8
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n))
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    return BLOCK_SIZE, num_warps

@triton.jit
def layernorm_forward(
        X,
        W,
        B,
        Y,
        r,
        mu,
        stride_x_row,
        stride_y_row,
        n,
        eps,
        BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n
    x_ptrs = X + row * stride_x_row + cols
    x = tl.load(x_ptrs, mask=mask, other=0.).to(tl.float32)
    x_centered = x - tl.max(x, axis=0)
    sq = x_centered * x_centered
    r_num = tl.sum(sq, axis=0)
    r_denom = r_num / n
    r = tl.sqrt(r_denom) + 1e-7
    r_inv = 1 / r
    y = x_centered * r_inv
    y_hat = y * r_num / n
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    Y_row = Y + row * stride_y_row
    tl.store(Y_row + cols, y_hat * w + b, mask=mask)
    tl.store(mu + row, y_hat.to(mu.dtype.element_ty), mask=True)
    tl.store(r + row, r.to(mu.dtype.element_ty), mask=True)

class layernorm_backward(triton.jit):
    def __init__(self):
        self.id = None

    @triton.jit
    def func(self,
            dY,
            X,
            W,
            B,
            Y,
            r,
            mu,
            dX,
            stride_x_row,
            stride_y_row,
            n,
            eps,
            BLOCK_SIZE: tl.constexpr, ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n
        x_ptrs = X + row * stride_x_row + cols
        x = tl.load(x_ptrs, mask=mask, other=0.).to(tl.float32)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        r = r + row
        mu = mu + row
        wdy = w * dY
        dy = (x - tl.load(mu)) * wdy
        dx = dy.to(tl.float32)
        var = tl.load(r)
        dvar = tl.sum(dx * (x - tl.load(mu)), axis=0)
        dmu = -0.5 * tl.sum((x - tl.load(mu)) / tl.sqrt(var + eps) * dvar, axis=0)
        dx += (x - tl.load(mu)) * dvar / tl.sqrt(var + eps) + dmu * 0
        w = tl.load(B + cols, mask=mask).to(tl.float32)
        Y_row = Y + row * stride_y_row
        dy = tl.load(Y_row + cols, mask=mask, other=0.).to(tl.float32)
        dX_row = dX + row * stride_x_row
        dY_row = dY + cols
        dW = dW + cols
        dB = dB + cols
        tl.store(dX_row + cols, dx, mask=mask)
        tl.store(dW, (wdy * (dy - tl.load(r) * tl.load(mu))).to(w.dtype), mask=mask)
        tl.store(dB, (wdy).to(w.dtype), mask=mask)

class Fast_Layernorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        x = x.contiguous()
        y = torch.empty_like(x)
        n = x.shape[-1]
        M, N = x.shape
        m = torch.empty((M, N, ), dtype=x.dtype, device=x.device)
        r = torch.empty((M, N, ), dtype=x.dtype, device=x.device)
        BLOCK_SIZE, num_warps = calculate_settings(n)
        BLOCK_SIZE = min(BLOCK_SIZE, 1024)
        layernorm_forward[(M, )](
            x, weight, bias, y, r, m,  #
            x.stride(0), y.stride(0), n, eps,  #
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        ctx.save_for_backward(x, weight, bias, r, m)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y

    @staticmethod
    def backward(ctx, dY):
        x, weight, bias, r, m = ctx.saved_tensors
        output_dtype = x.dtype.base_ty
        dX = torch.empty_like(x)
        dW = torch.empty((x.shape[-1], ), dtype=output_dtype, device=x.device)
        dB = torch.empty((x.shape[-1], ), dtype=output_dtype, device=x.device)
        layernorm_backward = layernorm_backward(dW, dB)
        layernorm_backward.func[(x.shape[0], )](
            dW, dB,  #
            dY, x, weight, bias, dX, r, m,  #
            dY.stride(0), x.stride(0), x.shape[-1], 1e-12,  #
            BLOCK_SIZE=ctx.BLOCK_SIZE)
        return dX, None, dW, dB, None

fast_layernorm = Fast_Layernorm.apply


import types

def fast_layernorm(input: Tensor, normalized_shape: Union[int, List[int]], weight: Optional[Tensor],
                   bias: Optional[Tensor], eps: float = 1.0e-5) -> Tensor:
    if isinstance(normalized_shape, int):
        pass
    else:
        raise RuntimeError("Only supporting a single vector norm for now")
    normalized_shape = input.shape[-1]
    with torch.cuda.device(input.device):

        if isinstance(weight, types.NoneType):
            weight = torch.ones(normalized_shape, device=input.device)
        if isinstance(bias, types.NoneType):
            bias = torch.zeros(normalized_shape, device=input.device)

        y = Fast_Layernorm.apply(input, normalized_shape, weight, bias, eps)
    return y
