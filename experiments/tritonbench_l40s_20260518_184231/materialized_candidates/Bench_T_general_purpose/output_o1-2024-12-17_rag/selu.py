import triton
import triton.language as tl
import torch

@triton.jit
def _selu_kernel(
    X, 
    OUT, 
    n_elements,
    alpha,
    scale,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X + offsets, mask=mask, other=0.0)
    x_pos = tl.where(x > 0.0, x, 0.0)
    x_neg = tl.where(x <= 0.0, alpha * (tl.exp(x) - 1.0), 0.0)
    y = scale * (x_pos + x_neg)
    tl.store(OUT + offsets, y, mask=mask)

def selu(input, inplace=False):
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    if inplace:
        out = input
    else:
        out = input.clone()
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _selu_kernel[grid](out, out, n_elements, alpha, scale, BLOCK_SIZE=BLOCK_SIZE)
    return out
