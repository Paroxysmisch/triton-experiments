import torch
import triton
import triton.language as tl

from xformers.components import Activation
from xformers.components.activations.triton_impl.activations import leaky_relu_grad

class LeakyReLU(Activation):
    r"""Applies the Leaky ReLU activation function element-wise to the input tensor.

    The function is defined as :math:`LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)`,
    where :math:`negative_slope` is a small constant that allows a small, non-zero gradient when the unit is not active.

    Args:
        negative_slope (float, optional): Controls the angle of the leaky ReLU. Default: 0.01
        inplace (bool, optional): If set to ``True``, the input tensor is replaced in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. _LeakyReLU: https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html

    Example:
        >>> m = LeakyReLU(0.1)
        >>> input = torch.Tensor([-1, 2])
        >>> output = m(input)
        >>> output
        tensor([-0.1000,  2.0000])
    """

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    @staticmethod
    @triton.jit
    def leaky_relu(x):
        scale = 0.01 + 0.0
        scale = scale.to(x.dtype)
        return tl.where(x >= 0, x, scale * x)

    @staticmethod
    @triton.jit
    def leaky_relu_grad(x):
        min_grad = 0.01
        max_grad = 1

        min_grad = min_grad.to(x.dtype)
        max_grad = max_grad.to(x.dtype)

        return tl.where(x >= 0, max_grad, min_grad)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            _ = torch.fmax(input, torch.tensor(0.0, dtype=input.dtype), out=input)
            input += self.negative_slope * torch.fmin(input, torch.tensor(0.0, dtype=input.dtype))
            return input
        else:
            return self.leaky_relu(input)

    def backward(self, input: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu_grad(input) * grad_output

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}, inplace={self.inplace}"
