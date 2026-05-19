import torch
import triton
import triton.language as tl

@triton.jit
def linear_elu_kernel(x_ptr, y_ptr, w_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    w = tl.load(w_ptr + offsets, mask=mask)
    b = tl.load(b_ptr)

    linear = x * w + b
    output = tl.where(linear > 0, linear, 0) + tl.where(linear <= 0, tl.exp(linear) - 1, 0)

    tl.store(y_ptr + offsets, output, mask=mask)

def linear_elu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    y = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda
    n_elements = y.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    linear_elu_kernel[grid](x, y, w, b, n_elements, BLOCK_SIZE=1024)

    return y
