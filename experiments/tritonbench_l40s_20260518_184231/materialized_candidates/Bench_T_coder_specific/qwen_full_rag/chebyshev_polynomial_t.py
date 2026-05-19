import torch
import triton
import triton.language as tl


@triton.jit
def _chebyshev_polynomial_t_recurrence(x0, x1, x):
    return 2 * x * x1 - x0


@triton.jit
def _chebyshev_polynomial_t_trig(x, n):
    return tl.cos(n * tl.acos(x))


@triton.jit
def _chebyshev_polynomial_t(x, n, out=None):
    if x.dtype.is_fp64():
        x = x.to(tl.float32)

    if out is None:
        out = tl.full(x.shape, 0.0, dtype=x.dtype)
    else:
        assert out.dtype == x.dtype

    cond_0 = n == 0
    cond_1 = n == 1
    cond_default = (n < 6) | (tl.abs(x) > 1.0)
    default = _chebyshev_polynomial_t_recurrence(1.0, x, x)
    ones = tl.full(x.shape, 1.0, dtype=x.dtype)
    zeros = tl.full(x.shape, 0.0, dtype=x.dtype)
    out = tl.where(cond_0, ones, out)
    out = tl.where(cond_1, x, out)
    out = tl.where(cond_default, default, out)
    out = out.to(x.dtype)
    return out


def chebyshev_polynomial_t(input, n, *, out=None):
    """
    Compute the Chebyshev polynomial of the first kind T_n(input).
    .. note::
        If n = 0, returns 1.
        If n = 1, returns input.
        For n < 6 or |input| > 1, uses a recursive formula.
        Otherwise, uses an explicit trigonometric formula.
    Args:
        input (Tensor): the input tensor.
        n (Tensor): Degree of the polynomial.
    Keyword args:
        out (Tensor, optional): the output tensor.
    Returns:
        Tensor
    """

    return _chebyshev_polynomial_t(input, n, out=out)
