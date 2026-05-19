import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import FunctionCtx
from torch._inductor.runtime.triton_heuristics import grid
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers


class BatchNormFunction(FunctionCtx):

    @staticmethod
    @triton.jit
    def forward_kernel(
        X,
        Y,
        W,
        B,
        Mean,
        Rstd,
        stride_x,
        stride_y,
        N,
        HW,
        C,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Calculate thread index
        row = tl.program_id(0)
        col = tl.program_id(1)
        # The code processes data in blocks
        for k in range(C):
            offs = (row * HW + col) * C + k
            # Locking mechanism to ensure thread safety when accessing shared variables
            with tl.lock(offs):
                # Calculate mean
                _mean = tl.sum(tl.load(X + offs) for _ in range(BLOCK_SIZE)) / BLOCK_SIZE
                if row == 0:
                    # Store mean
                    tl.store(Mean + k, _mean)
                # Synchronization barrier
                tl.debug_barrier()
                # Calculate variance
                _var = tl.sum((tl.load(X + offs) - _mean) ** 2 for _ in range(BLOCK_SIZE)) / BLOCK_SIZE
                if row == 0:
                    # Store variance
                    inv_std = 1 / tl.sqrt(_var + eps)
                    tl.store(Rstd + k, inv_std)
                # Synchronization barrier
                tl.debug_barrier()
            # Calculate normalization factor
            x_hat = (tl.load(X + offs) - tl.load(Mean + k)) * tl.load(Rstd + k)
            # Apply affine transformation
            y = x_hat * tl.load(W + k) + tl.load(B + k)
            # Write output
            tl.store(Y + offs, y)

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor, training: bool, momentum: float, eps: float):
        ctx.save_for_backward(x, weight, bias)
        ctx.eps = eps

        x = x.contiguous()
        dim = x.shape[-1]
        N = x.numel() // dim
        shape_out = x.shape
        x = x.view(N, dim)
        mean = torch.zeros(dim, dtype=torch.float32, device=x.device)
        rstd = torch.ones(dim, dtype=torch.float32, device=x.device)
        out = torch.empty_like(x)

        if training:
            mean, rstd = moving_statistics(x, momentum=momentum, eps=eps, out=(mean, rstd))
        else:
            mean, rstd = running_mean.clone(), running_var.rsqrt().clone()

        x_hat = (x - mean) * rstd
        out = x_hat * weight[:, None] + bias[:, None]

        out = out.view(shape_out)
        ctx.mean, ctx.rstd = mean, rstd
        return out


def batch_norm(x: Tensor, weight: Tensor, bias: Tensor, running_mean: Tensor, running_var: Tensor, training: bool, momentum: float, eps: float) -> Tensor:
    assert x.is_contiguous()
    ndim = x.ndim
    assert 2 <= ndim <= 4, "Input must between 2D and 4D"
    x_strides = x.stride()
    dim = x.size(-1)
    N = x.numel() // dim
    shape_i = x.shape[:-1]
    x = x.reshape(N, dim)
    out = torch.empty_like(x)

    BLOCK_SIZE = triton_helpers.next_power_of_2(dim)
    grid_fn = grid(meta={'BLOCK_SIZE': BLOCK_SIZE})
    desc = instance_descriptor(divisible_by_16=ndim - 2)

    grid_fn[(shape_i[0],)](BatchNormFunction.forward_kernel,
                           x,
                           out,
                           weight,
                           bias,
                           running_mean,
                           running_var,
                           x_strides[0],
                           out.stride(0),
                           N,
                           shape_i[1],
                           dim,
                           BLOCK_SIZE=BLOCK_SIZE,
                           num_warps=4,
                           tile_hint=TileHint.SQUARE,
                           descriptor=desc,
                           min_num_programs=1)

    return out
