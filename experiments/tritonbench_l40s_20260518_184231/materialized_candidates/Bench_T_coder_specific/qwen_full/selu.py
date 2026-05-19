import torch
import triton
import triton.language as tl
from triton.language.libdevice import exp2

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def swiglu(x):
    return silu(x[:, :, :, :self.c // 2]) * x[:, :, :, self.c // 2:self.c]

@triton.jit
def relu(x):
    return tl.maximum(0, x)

@triton.jit
def softplus(x):
    return tl.math.log(tl.math.exp(x) + 1)

@triton.jit
def gelu_new(x):
    pi = 3.141592653589793
    a = tl.math.sqrt(2.0 / pi)
    b = x + 0.044715 * x * x * x
    return 0.5 * x * (1.0 + softplus(a * b))

@triton.jit
def div_rn(x, y):
    return tl.math.fdiv(x, y)

@triton.jit
def rsqrt(x):
    return tl.math.rsqrt(x)

@triton.jit
def norm(x, axis):
    return tl.math.sqrt(tl.sum(x * x, axis))

@triton.jit
def softmax(x, ieee_rounding=False):
    z = x - tl.max(x, 0)
    num = tl.exp(z)
    den = tl.sum(num, 0)
    return div_rn(num, den)

@triton.jit
def dropout(x, p, seed, offset):
    random = fast_rand(seed, offset, 0, 2**32 - 1)
    return tl.where(random > p * 2**32, x / (1 - p), 0.0)

@triton.jit
def selu(x):
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return scale * (relu(x) + alpha * exp2(x - 1))
