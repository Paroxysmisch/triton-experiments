import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    X,
    W,
    Y,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * stride
    X += pid * stride

    mask = tl.arange(0, BLOCK_SIZE) < N

    x = tl.load(X + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    var = tl.sum(x * x, axis=0) / N
    rrms = tl.math.rsqrt(var + eps)

    y = (x * rrms).to(Y.dtype.element_ty) * w

    tl.store(Y + tl.arange(0, BLOCK_SIZE), y, mask=mask)


class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-6):
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = triton.next_power_of_2(N)
        num_warps = 4

        y = torch.empty_like(x)

        rms_norm_kernel[M,](
            x,
            weight,
            y,
            x.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.eps = eps
        ctx.save_for_backward(x, weight, y)
        return y


def rms_norm(x, normalized_shape, weight, eps=1e-6):
    return RmsNorm.apply(x, normalized_shape, weight, eps)
