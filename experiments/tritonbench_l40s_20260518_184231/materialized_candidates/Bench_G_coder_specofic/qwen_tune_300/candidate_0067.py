import torch
import triton
import triton.language as tl
from torch.autograd.function import Function

@triton.jit
def rms_norm_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    X += pid * stride
    Y += pid * stride
    w = tl.load(W + tl.arange(0, BLOCK_SIZE))

    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N

    # Compute rrms (reciprocal of the root of the variance)
    rrms = 1 / tl.sqrt(var + eps)

    # Normalize and apply linear transformation
    for i in range(0, N, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x * rrms).to(Y.dtype.element_ty) * w
        tl.store(Y + cols, y, mask=mask)

class RmsNorm(Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = triton.next_power_of_2(N)
        x = x.contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x)

        with torch.cuda.device(x.device):
            rms_norm_kernel[M,](
                x,
                y,
                weight,
                x.stride(dim),
                N,
                eps,
                BLOCK_SIZE,
            )
        return y

def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)
