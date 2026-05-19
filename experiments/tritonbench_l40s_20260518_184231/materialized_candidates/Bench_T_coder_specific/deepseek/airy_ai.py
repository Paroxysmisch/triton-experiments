(Writing code here)

def airy_ai(input, *, out=None):
    return triton.ops.airy_ai(input, out=out)

#src/triton/python/ops/triton/ops/__init__.py
from . import nn
from . import linalg
from . import functional
from . import random
from . import sort
from . import math
from . import linalg
from . import _triton import *

from .math import *
from .nn import *
from .linalg import *
from .functional import *
from .random import *
from .sort import *
from .linalg import *

#src/triton/python/ops/triton/ops/random.py
from . import _triton


def randn(shape, *, dtype=None, device=None):
    """
    Returns a tensor of the specified shape and dtype, filled with random numbers
    from a normal distribution with mean 0 and standard deviation 1.
    """
    return _triton.randn(shape, dtype=dtype, device=device)


def rand(shape, *, dtype=None, device=None):
    """
    Returns a tensor of the specified shape and dtype, filled with random numbers
    from a uniform distribution over [0, 1).
    """
    return _triton.rand(shape, dtype=dtype, device=device)

#src/triton/python/ops/triton/ops/sort.py
from . import _triton


def argsort(input, *, dim=-1, descending=False, out=None):
    """
    Returns the indices that sort the elements of the input tensor along a given
    dimension in ascending order.
    """
    return _triton.argsort(input, dim=dim, descending=descending, out=out)


def sort(input, *, dim=-1, descending=False, out=None):
    """
    Returns a new tensor with the elements of the input tensor in the
    given dimension sorted in ascending order.
    """
    return _triton.sort(input, dim=dim, descending=descending, out=out)

#src/triton/python/ops/triton/ops/utils.py
import numpy as np
from . import _triton


def to_numpy(tensor):
    """
    Converts a Tensor to a numpy array.
    """
    return np.asarray(_triton.to_numpy(tensor))


def from_numpy(array):
    """
    Converts a numpy array to a Tensor.
    """
    return _triton.from_numpy(array)

#src/triton/python/ops/triton/ops/nn.py
from . import _triton
from .utils import to_numpy


def conv2d(input, weight, *, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    """
    Applies a 2D convolution over an input signal composed of several input
    planes.
    """
    return _triton.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups, out=out)


def linear(input, weight, *, bias=None, out=None):
    """
    Applies a linear transformation to the input: :math:`y = xA^T + b`.
    """
    return _triton.linear(input, weight, bias=bias, out=out)


def relu(input, *, out=None):
    """
    Applies the rectified linear unit function element-wise:
    :math:`y = max(0, x)`.
    """
    return _triton.relu(input, out=out)


def max_pool2d(input, *, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, out=None):
    """
    Applies a 2D max pooling over an input signal composed of several input
    planes.
    """
    if stride is None:
        stride = kernel_size
    return _triton.max_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode, out=out)


def avg_pool2d(input, *, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, out=None):
    """
    Applies a 2D average pooling over an input signal composed of several input
    planes.
    """
    if stride is None:
        stride = kernel_size
    return _triton.avg_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, out=out)


def dropout(input, *, p=0.5, training=True, inplace=False, out=None):
    """
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.
    """
    return _triton.dropout(input, p=p, training=training, inplace=inplace, out=out)


def batch_norm(input, *, weight=None, bias=None, running_mean=None, running_var=None, training=True, momentum=0.1, eps=1e-5, out=None):
    """
    Applies Batch Normalization over a 4D input as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift`_ .
    """
    return _triton.batch_norm(input, weight=weight, bias=bias, running_mean=running_mean, running_var=running_var, training=training, momentum=momentum, eps=eps, out=out)


def log_softmax(input, *, dim=-1, out=None):
    """
    Applies the log softmax function to an n-dimensional input Tensor.
    The log_softmax function is defined as follows:
    :math:`\text{log_softmax}(x_{i}) = x_{i} - log(\sum_{j} exp(x_{j}))`
    """
    return _triton.log_softmax(input, dim=dim, out=out)


def softmax(input, *, dim=-1, out=None):
    """
    Applies the softmax function to an n-dimensional input Tensor.
    The softmax function is defined as follows:
    :math:`\text{softmax}(x_{i}) = exp(x_{i}) / \sum_{j} exp(x_{j})`
    """
    return _triton.softmax(input, dim=dim, out=out)

#src/triton/python/ops/triton/ops/functional.py
from . import _triton
from .utils import to_numpy


def sigmoid(input, *, out=None):
    """
    Applies the sigmoid function element-wise:
    :math:`\text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}`.
    """
    return _triton.sigmoid(input, out=out)


def tanh(input, *, out=None):
    """
    Applies the hyperbolic tangent function element-wise:
    :math:`\text{tanh}(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}`.
    """
    return _triton.tanh(input, out=out)


def relu6(input, *, out=None):
    """
    Applies the element-wise function:
    :math:`\text{ReLU6}(x) =
