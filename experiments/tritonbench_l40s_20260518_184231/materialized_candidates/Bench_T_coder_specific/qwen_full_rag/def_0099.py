import logging
import math
import paddle
import triton
import triton.language as tl
from triton.language.libdevice import erf, pow, tanh

@triton.jit
def gelu_none_and_reduce_kernel(x, w, dim_size, stride, N):
    # Convert inputs to float32 for better precision
    x_fp32 = x.to(tl.float32)
    # Compute GELU using the error function approximation
    x_gelu = 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811))
    # Normalize and compute the mean square error
    x_centered = x_gelu - tl.sum(x_gelu) / dim_size
    x_hat = tl.where(
        tl.arange(0, N) % stride == 0, x_centered * x_centered, 0
    ).to(tl.float32)
    return tl.sum(x_hat) / max(0, dim_size - 1)

@triton.jit
def gelu_tanh_and_reduce_kernel(x, w, dim_size, stride, N):
    # Convert inputs to float32 for better precision
    x_fp32 = x.to(tl.float32)
    # Compute GELU using the tanh approximation
    x_gelu = (
        0.5
        * x_fp32
        * (1 + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32.to(tl.float32), 2)))))
    )
    # Normalize and compute the mean square error
    x_centered = x_gelu - tl.sum(x_gelu) / dim_size
    x_hat = tl.where(
        tl.arange(0, N) % stride == 0, x_centered * x_centered, 0
    ).to(tl.float32)
    return tl.sum(x_hat) / max(0, dim_size - 1)

class GeLUStd(paddle.autograd.Function):
    @staticmethod
    def forward(ctx, A, dim=None, keepdim=False, correction=1, approximate="none"):
        logging.debug("TRITON GELU STD FORWARD")
        if dim is None:
            out = paddle.reshape(A, [-1])
            dim = 0
        else:
            shape = list(A.shape)
            if isinstance(dim, int):
                dim = [dim]
            for d in dim:
                if d < -A.ndim or d >= A.ndim:
                    raise IndexError(
                        "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                            -A.ndim, A.ndim - 1, d
                        )
                    )
            assert len(shape) > 0, "reduce over zero-size dimension"
            dim = sorted(dim)
            ndim = len(shape)
            dim_size = 1
            stride = 1
            for i in range(ndim - 1, -1, -1):
                if i in dim:
                    dim_size *= shape[i]
                else:
                    if shape[i] != 1:
                        stride *= shape[i]
            out_shape = list(A.shape)
            for i in dim:
                out_shape[i] = 1
            if not keepdim:
                for i in dim:
                    out_shape.pop(i)
                out = paddle.reshape(A, out_shape)
            else:
                out = A
            N = out.size(0)
            vec = paddle.reshape(out, [N])
            if approximate == "none":
                variance = gelu_none_and_reduce_kernel(vec, None, dim_size, stride, N)
            elif approximate == "tanh":
                variance = gelu_tanh_and_reduce_kernel(vec, None, dim_size, stride, N)
            else:
                raise ValueError(f"Invalid approximate value: {approximate}")
            ctx.save_for_backward(A)
            ctx.dim = dim
            ctx.keepdim = keepdim
            ctx.correction = correction
            ctx.approximate = approximate
            std = paddle.sqrt(variance * correction)
            return std
        return out

def gelu_std(A, dim=None, keepdim=False, correction=1, approximate="none"):
    return GeLUStd.apply(A, dim, keepdim, correction, approximate)
