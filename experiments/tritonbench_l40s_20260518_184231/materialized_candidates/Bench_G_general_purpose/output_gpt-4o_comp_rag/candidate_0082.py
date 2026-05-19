import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_fused(X, Y, W, B, Mean, Rstd, stride_ml, stride_n, L, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    batch = tl.program_id(1)
    base_idx = row * stride_ml + batch * stride_n
    Y += base_idx
    X += base_idx

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
        _var += a * a

    mean = tl.sum(_mean) / N
    var = tl.sum(_var) / N - mean * mean
    rstd = 1 / tl.sqrt(var + eps)

    tl.store(Mean + row * L + batch, mean)
    tl.store(Rstd + row * L + batch, rstd)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def _layer_norm_bwd_dx_fused(input_ptr, weight_ptr, grad_output_ptr, mean_ptr, rstd_ptr, grad_input_ptr, grad_weight_accum_ptr, grad_bias_accum_ptr, stride_ml, stride_n, L, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    batch = tl.program_id(1)
    base_idx = row * stride_ml + batch * stride_n
    grad_input_ptr += base_idx
    input_ptr += base_idx
    weight_ptr += base_idx
    grad_output_ptr += base_idx

    mean = tl.load(mean_ptr + row * L + batch)
    rstd = tl.load(rstd_ptr + row * L + batch)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(input_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask)
        dy = tl.load(grad_output_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        x_hat = (x - mean) * rstd
        dx_hat = dy * w
        dx = (dx_hat - tl.sum(dx_hat) / N - x_hat * tl.sum(dx_hat * x_hat) / N) * rstd
        tl.store(grad_input_ptr + cols, dx, mask=mask)

        dw = dy * x_hat
        db = dy
        tl.atomic_add(grad_weight_accum_ptr + cols, dw, mask=mask)
        tl.atomic_add(grad_bias_accum_ptr + cols, db, mask=mask)

@triton.jit
def _layer_norm_bwd_dwdb(grad_weight_accum_ptr, grad_bias_accum_ptr, final_grad_weight_ptr, final_grad_bias_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    for off in range(0, num_elements, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < num_elements
        dw = tl.load(grad_weight_accum_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        db = tl.load(grad_bias_accum_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.atomic_add(final_grad_weight_ptr + cols, dw, mask=mask)
        tl.atomic_add(final_grad_bias_ptr + cols, db, mask=mask)

class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        M, L, N = x.shape
        BLOCK_SIZE = 128
        ctx.eps = eps
        ctx.M, ctx.L, ctx.N = M, L, N

        y = torch.empty_like(x)
        mean = torch.empty((M, L), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, L), dtype=torch.float32, device=x.device)

        grid = lambda meta: (M, L)
        _layer_norm_fwd_fused[grid](x, y, weight, bias, mean, rstd, x.stride(0), x.stride(1), L, N, eps, BLOCK_SIZE)

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        M, L, N = ctx.M, ctx.L, ctx.N
        eps = ctx.eps
        BLOCK_SIZE = 128

        dx = torch.empty_like(x)
        grad_weight_accum = torch.zeros_like(weight)
        grad_bias_accum = torch.zeros_like(bias)

        grid = lambda meta: (M, L)
        _layer_norm_bwd_dx_fused[grid](x, weight, dy, mean, rstd, dx, grad_weight_accum, grad_bias_accum, x.stride(0), x.stride(1), L, N, eps, BLOCK_SIZE)

        final_grad_weight = torch.zeros_like(weight)
        final_grad_bias = torch.zeros_like(bias)
        _layer_norm_bwd_dwdb[(N + BLOCK_SIZE - 1) // BLOCK_SIZE](
            grad_weight_accum, grad_bias_accum, final_grad_weight, final_grad_bias, N, BLOCK_SIZE
        )

        return dx, final_grad_weight, final_grad_bias, None

def layer_norm(x, weight, bias, eps=1e-5):
    return LayerNorm.apply(x, weight, bias, eps)
