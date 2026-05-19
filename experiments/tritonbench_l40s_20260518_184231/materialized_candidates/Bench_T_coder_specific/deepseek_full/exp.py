import torch
import triton
import triton.language as tl

@triton.jit
def exp(x):
    # Math: y_i = e^x_i
    return tl.exp(x)

def exp(input, *, out=None):
    # type: (Tensor, *, Tensor) -> Tensor
    r"""Returns a new tensor with the exponential of the elements of the input tensor.

    .. math::
        \text{out}_i = e^{\text{input}_i}

    Args:
        input (Tensor): the input tensor.
        out (Tensor, optional): the output tensor.

    Returns:
        Tensor: the exponential of the input tensor.

    Example::

        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> torch.exp(x)
        tensor([ 2.7183,  7.3891, 20.0855, 54.5982])
    """
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32, device=input.device)
    input_arg = input
    if input_arg.dtype is torch.bool:
        input_arg = input_arg.to(torch.int64)
    exp(input_arg, out=out)
    if input.dtype is torch.bool:
        out.to(torch.bool, copy=True, non_blocking=True)
    return out
