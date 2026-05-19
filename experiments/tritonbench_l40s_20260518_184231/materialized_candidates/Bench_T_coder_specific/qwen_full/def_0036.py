import torch
import triton
import triton.language as tl
from triton.language.libdevice import erf, exp, pow, tanh

@triton.jit
def gelu_functor_exact(x):
    # Using approximation can improve performance
    scale: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    return x * 0.5 * (1.0 + erf(x * scale))

@triton.jit
def gelu_functor_tanh(x):
    # Using approximation can improve performance
    sqrt_2_over_pi: tl.constexpr = 0.7978845608  # math.sqrt(2 / math.pi)
    c: tl.constexpr = 0.044715  # math.sqrt(2) / 8
    return 0.5 * x * (1.0 + tanh(sqrt_2_over_pi * x * (1.0 + c * pow(x, 2))))

@triton.jit
def add_gelu_kernel(x_ptr, y_ptr, alpha, out_ptr, n_elements,
                    BLOCK_SIZE: tl.constexpr, approximate: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    other = tl.load(y_ptr + offsets, mask=mask)
    if approximate == 'none':
        x = gelu_functor_exact(x + alpha * other)
    else:
        x = gelu_functor_tanh(x + alpha * other)
    tl.store(out_ptr + offsets, x, mask=mask)

def add_gelu(input, other, alpha=1, approximate='none', out=None) -> torch.Tensor:
    assert input.is_contiguous()
    assert other.is_contiguous()
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_contiguous()
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    approximate = approximate.lower()
    assert approximate in ['none', 'tanh']
    with torch.cuda.device(input.device):
        add_gelu_kernel[grid](input, other, alpha, out, n_elements,
                              BLOCK_SIZE=1024, approximate=approximate)
    return out
