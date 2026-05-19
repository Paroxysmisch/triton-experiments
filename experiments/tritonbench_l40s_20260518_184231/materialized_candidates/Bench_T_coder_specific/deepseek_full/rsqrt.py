import torch
import triton
import triton.language as tl

@triton.jit
def rsqrt(input):
    # Compute the reciprocal of the square-root of each element
    output = tl.math.rsqrt(input)
    return output

def rsqrt(input, *, out=None):
    # type: (Tensor, *, Tensor) -> Tensor
    r"""Returns a new tensor with the reciprocal of the square-root of each of the elements of the input tensor.

    .. math::
        \text{out}_{i} = \frac{1}{\sqrt{\text{input}_{i}}}

    Note:
        The function will return 'nan' for negative input values.

    Args:
        input (Tensor): the input tensor.

    Keyword args:
        out (Tensor, optional): the output tensor.

    Example:
        >>> a = torch.tensor([1.0, 2.0, 3.0, 4.0], device='triton')
        >>> torch.rsqrt(a)
        tensor([1.0000, 0.7071, 0.5774, 0.5000], device='triton')
    """
    if out is None:
        out = torch.empty_like(input)

    rsqrt[input.shape](input, out=out)

    return out
