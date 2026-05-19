import torch
import triton
import triton.language as tl
from triton.language.math import asin as _tl_asin


def asin(input, *, out=None):
    """
    asin(input, *, out=None) -> Tensor

    Return a new tensor with the arcsine of the elements of :attr:`input`.

    .. math::
        \\text{out}_i = \\sin^{-1}(\\text{input}_i)

    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor

    Returns:
        Tensor: the output tensor

    Example:

        >>> input = torch.tensor([0.5, -0.5, 1.0], dtype=torch.float32)
        >>> torch.asin(input)
        tensor([0.5236, -0.5236,  1.5708], dtype=torch.float32)
    """
    if out is None:
        out = torch.empty_like(input, dtype=input.dtype)
    elif out.dtype != input.dtype:
        raise RuntimeError("input and out must have the same dtype")
    if input.is_floating_point():
        assert (
            input.dim() == out.dim()
        ), "Only scalar or broadcastable inputs are allowed"
        inp_shape = list(input.shape)
        out_strides = list(out.stride())
        out_shape = [prod(out_strides)]
        for i, s in enumerate(inp_shape):
            if s != 1 and out_shape[-1] % s != 0:
                out_shape.append(s)
                out_shape.extend(1 for _ in out_strides)
        stride = 1
        out_strides = []
        for s in reversed(out_shape):
            if s == 1:
                out_strides.append(0)
            else:
                out_strides.append(stride)
                stride *= s
        out_strides.reverse()
        out = out.reshape(out_shape).as_strided(out_shape, out_strides)
        if input.numel() > 0:
            _tl_asin(
                input,
                out,
                input.numel(),
            )
        return out
    else:
        return torch.full(input.shape, float("nan"), dtype=input.dtype, device=input.device)
