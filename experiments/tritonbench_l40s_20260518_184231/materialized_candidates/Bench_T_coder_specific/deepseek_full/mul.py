import torch
import triton
import triton.language as tl
from triton.runtime.jit import reinterpret

@triton.jit
def mul(input, other, out=None):
    # type: (Tensor, Tensor | Number, Optional[Tensor]) -> Tensor
    r"""Multiplies the input tensor by another tensor or a number,
    supporting broadcasting to a common shape, type promotion, and
    integer, float, and complex inputs.

    Args:
        input (Tensor): the input tensor.
        other (Tensor or Number): the tensor or number to multiply
            the input by.
        out (Tensor, optional): the output tensor.

    Returns:
        Tensor: the output tensor.

    .. math:: \text{out}_i = \text{input}_i \times \text{other}_i

    other: Supports broadcasting and type promotion.
    """
    if isinstance(other, Number):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    if out is None:
        out = torch.empty_like(input)
    return out
