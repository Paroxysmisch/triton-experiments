import torch
import triton
import triton.language as tl

@triton.jit
def _add(input, other, alpha):
    return input + other * alpha

def add(input, other, *, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape and out.dtype == input.dtype
    if isinstance(other, torch.Tensor):
        assert other.shape broadcastable_with input.shape
        other = other.to(input.dtype)
    elif isinstance(other, (int, float, complex)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    else:
        raise TypeError("Unsupported other type: {}".format(type(other)))
    return _add(input, other, alpha, out=out)
