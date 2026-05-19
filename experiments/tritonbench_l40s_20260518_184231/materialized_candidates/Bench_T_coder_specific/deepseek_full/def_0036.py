import triton
import triton.language as tl
from torch import Tensor
from .gaussian_error_linear_unit import gelu_exact, gelu_tanh

@triton.jit
def add_gelu(input, other, alpha, approximate, **meta):
    idx = tl.arange(0, meta['N'])
    input_tensor = tl.load(input + idx)
    if isinstance(other, (Tensor, triton.language.views.ContiguousView)):
        other_tensor = tl.load(other + idx)
    else:
        other_tensor = other
    result = input_tensor + alpha * other_tensor
    if approximate == 'none':
        result = gelu_exact(result)
    elif approximate == 'tanh':
        result = gelu_tanh(result)
    else:
        raise ValueError("Invalid value for 'approximate'")
    tl.store(out + idx, result)

def add_gelu_wrapper(input, other, alpha=1, approximate='none', out=None):
    if out is None:
        out = input
    elif not isinstance(out, Tensor):
        raise TypeError("'out' must be a Tensor or None")
    if approximate not in ['none', 'tanh']:
        raise ValueError("Invalid value for 'approximate'")
    if not isinstance(input, Tensor):
        raise TypeError("'input' must be a Tensor")
    if isinstance(other, Tensor) and other.shape != input.shape:
        raise ValueError("shapes of input and other must be the same")
    if isinstance(alpha, Tensor):
        raise TypeError("'alpha' cannot be a Tensor")
    if out.shape != input.shape:
        raise ValueError("shapes of input and out must be the same")
    meta = {'N': input.numel(), 'approximate': approximate}
    add_gelu[(input, other, alpha, approximate, out)](meta=meta)
    return out
