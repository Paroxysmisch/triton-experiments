import torch
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

def tanh(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_floating_point()
    if input.requires_grad or out.requires_grad:
        out.copy_(tanh_kernel(input))
    else:
        out.copy_(tanh_kernel(input), non_blocking=True)
    return out
