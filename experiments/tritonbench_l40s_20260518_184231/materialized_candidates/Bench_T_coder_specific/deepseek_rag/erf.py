import torch
import triton
import triton.language as tl
from deepspeed.accelerator import get_accelerator

@triton.jit
def gelu_functor(x):
    # Using approximation introduces greater parity errors.
    # return tl.sigmoid(1.702 * x) * x
    return x * 0.5 * (1.0 + tl.libdevice.erf(x / 1.41421356237))

@triton.jit
def gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = gelu_functor(x)
    tl.store(output_ptr + offsets, output, mask=mask)

def gelu(activations: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(activations)
    else:
        assert out.is_contiguous() and get_accelerator().on_accelerator(out)

    assert activations.is_contiguous() and get_accelerator().on_accelerator(activations)

    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    gelu_kernel[grid](activations, out, n_elements, BLOCK_SIZE=1024)
    return out
