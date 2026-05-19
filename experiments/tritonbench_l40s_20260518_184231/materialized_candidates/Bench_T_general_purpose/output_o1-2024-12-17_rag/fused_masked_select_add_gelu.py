import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh, pow

@triton.jit
def _add_gelu_none_kernel(x_ptr, other_ptr, alpha_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_fp32 = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    other_fp32 = tl.load(other_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    alpha_val = tl.load(alpha_ptr)  # scalar alpha
    s = x_fp32 + alpha_val * other_fp32
    # GELU (approx='none') using erf
    gelu_val = 0.5 * s * (1.0 + erf(s * 0.7071067811))
    tl.store(out_ptr + offsets, gelu_val, mask=mask)

@triton.jit
def _add_gelu_tanh_kernel(x_ptr, other_ptr, alpha_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_fp32 = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    other_fp32 = tl.load(other_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    alpha_val = tl.load(alpha_ptr)  # scalar alpha
    s = x_fp
