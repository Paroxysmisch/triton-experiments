import torch
import triton

@triton.jit
def div(a, b, rounding_mode=None):
    if rounding_mode is None:
        return a / b
    else:
        raise NotImplementedError(
            "Only 'trunc' rounding mode is currently supported by the div op"
        )

def div(input, other, *, rounding_mode=None, out=None):
    triton_inp = input.is_contiguous()
    triton_other = other.is_contiguous() if isinstance(other, torch.Tensor) else True
    if triton_inp and triton_other:
        if input.ndim == 0 or other.ndim == 0:
            if input.ndim == 0:
                input = input.reshape((1,))
            if other.ndim == 0:
                other = other.reshape((1,))
            return div(input, other, rounding_mode=rounding_mode)
        if input.shape == other.shape:
            return div(input, other, rounding_mode=rounding_mode)
        if out is None:
            out = torch.empty_like(input)
        return div(input, other, rounding_mode=rounding_mode, out=out)
    else:
        return torch.div(input, other, out=out)
