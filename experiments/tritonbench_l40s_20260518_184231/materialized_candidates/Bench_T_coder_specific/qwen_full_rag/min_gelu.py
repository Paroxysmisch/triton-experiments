import logging
import math
import paddle
import paddle.triton
import paddle.triton.language as tl
from ..utils.shape_utils import _prod
from .min import min as paddle_min
from .gelu import gelu
from .tanh import tanh

@paddle.triton.jit
def min_gelu_none_and_mul_kernel(X, Y):
    X_fp32 = X.to(tl.float32)
    X_gelu = 0.5 * X_fp32 * (1 + tl.sigmoid(X_fp32 * 0.7071067811))
    return X_gelu * Y

@paddle.triton.jit
def min_gelu_tanh_and_mul_kernel(X, Y):
    X_fp32 = X.to(tl.float32)
    sqrt_2_over_pi = 0.7978845608028654
    coefficient = 0.04471505501042012
    X_gelu = 0.5 * X_fp32 * (
        1
        + tl.tanh(
            X_fp32 * sqrt_2_over_pi * (1 + coefficient * tl.pow(X_fp32, 2))
        )
    )
    return X_gelu * Y

class MinGeluAndMul(paddle.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, approximate="none"):
        logging.debug("MINIMUM GELU AND MUL FORWARD")
        if approximate == "none":
            return min_gelu_none_and_mul_kernel(A, B)
        elif approximate == "tanh":
            return min_gelu_tanh_and_mul_kernel(A, B)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")

def min_gelu_and_mul(A, B, approximate="none"):
    return MinGeluAndMul.apply(A, B, approximate)

def min_gelu(
    input,
    dim=None,
    keepdim=False,
    approximate="none",
    out=None,
):
    logging.debug("GEMS MINIMUM GELU")

    shape = input.shape
    if dim is None:
        input = paddle.reshape(input, [-1])
        dim = 0
    else:
        if dim < -input.ndim or dim >= input.ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -input.ndim, input.ndim - 1, dim
                )
            )

    input_shape_dim = shape[dim]
    shape[dim] = 1

    def meta_fn(x):
        return lambda meta: meta["X_BLOCK_SIZE"] > x

    increment = 1
    if _prod(shape) == 0:
        increment = 0
    elif out is None:
        num_warps = paddle.triton.next_power_of_2(input_shape_dim)
        intermediate = min_gelu_and_mul(
            input, paddle.full(shape, 1.0, dtype=input.dtype), approximate=approximate
        )
        if keepdim:
            return paddle_min(intermediate, dim=dim, keepdim=True)
        else:
            return paddle.squeeze(paddle_min(intermediate, dim=dim), axis=dim)
    else:
        assert out is not None
        assert out.shape == input.shape
        num_warps = paddle.triton.next_power_of_2(input_shape_dim)
        paddle_min(
            input,
            paddle.full(shape, float("-inf"), dtype=out.dtype),
            dim=dim,
            keepdim=keepdim,
            out=out,
            approximate=approximate,
        )
        return out
