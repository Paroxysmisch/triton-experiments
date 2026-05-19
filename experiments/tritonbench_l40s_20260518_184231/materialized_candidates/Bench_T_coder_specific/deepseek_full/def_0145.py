import triton
import triton.language as tl
from triton import Tensor

@triton.jit
def polygamma(n, input, *, out=None):
    # Compute the n-th derivative of the digamma function.
    # Implemented only for nonnegative integers n >= 0.
    #
    # Args:
    #     n (int): the order of the polygamma function
    #     input (Tensor): the input tensor
    #     out (Tensor, optional): the output tensor
    #
    # Returns:
    #     Tensor: the n-th derivative of the digamma function of the input tensor
    #         if out is None, otherwise out
    pass
