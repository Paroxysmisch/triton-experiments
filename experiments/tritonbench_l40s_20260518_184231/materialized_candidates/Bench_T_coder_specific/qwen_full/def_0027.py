import torch
import triton
import triton.language as tl

@triton.jit
def sqrt_tanh_kernel(x, y, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_values = tl.load(x + offsets, mask=mask)
    y_values = tl.sqrt(x_values)
    y_values = tl.math.tanh(y_values)
    tl.store(y + offsets, y_values, mask=mask)

def sqrt_tanh(input, out=None) -> torch.Tensor:
    assert input.is_contiguous()
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_contiguous()
    assert out.shape == input.shape

    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    sqrt_tanh_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
