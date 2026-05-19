import triton.language as tl
from xformers.components import Activation

@triton.jit
def leaky_relu(x, negative_slope=0.01):
    """
    LeakyReLU_ activation

    .. _LeakyReLU: https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    """
    zero = 0.0
    negative_slope = negative_slope.to(x.dtype)
    return tl.where(x >= 0, x, negative_slope * x)
