import torch
import triton
import triton.language as tl
from triton.language.libdevice import erf, exp, pow, tanh

@triton.jit
def _triton_gelu(x, approximate: tl.constexpr):
    if approximate == "tanh":
        # 0.5 * x * (1 + Tanh(√(2/π) * (x + 0.044715 * x^3)))
        return 0.5 * x * (1 + tanh(tl.sqrt(2.0 / tl.math.pi()) * (x + 0.044715 * pow(x, 3)))
    else:
        # x * Φ(x)
        return x * 0.5 * (1.0 + erf(x / tl.sqrt(2.0)))

def sub_gelu(input, other, alpha=1, approximate="none", out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_cuda and input.is_cuda and other.is_cuda

    assert input.is_cuda and other.is_cuda
    assert input.dim() > 0 and input.size(0) >= 1024

    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _triton_sub_gelu[grid](input, other, alpha, approximate, out, n_elements)

    return out

@triton.jit
def _triton_sub_gelu(input, other, alpha, approximate: tl.constexpr, out, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    input_ptrs = input + offset
    other_ptrs = other + offset
    input_vals = tl.load(input_ptrs, mask=mask)
    other_vals = tl.load(other_ptrs, mask=mask)
    output_vals = input_vals - alpha * other_vals
    output_vals = _triton_gelu(output_vals, approximate)
    tl.store(out + offset, output_vals, mask=mask)
