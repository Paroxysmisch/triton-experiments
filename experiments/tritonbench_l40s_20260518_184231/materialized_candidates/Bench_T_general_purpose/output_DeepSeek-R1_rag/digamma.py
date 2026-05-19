import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def digamma_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    # Compute digamma using libdevice and handle x=0
    y = libdevice.digamma(x)
    y = tl.where(x == 0.0, -tl.math.inf, y)
    tl.store(output_ptr + offsets, y, mask=mask)

def digamma(input, *, out=None):
    if not input.is_cuda:
        raise RuntimeError("Input tensor must be on CUDA device.")
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input.")
        if out.device != input.device:
            raise ValueError("Output tensor must be on the same device as input.")
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    digamma_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

# Validation
x = torch.tensor([0.0, 1.0, 2.0, 3.0], device='cuda')
out_triton = digamma(x)
out_torch = torch.digamma(x)
print("PyTorch:", out_torch)
print("Triton:", out_triton)
print(f'Max difference: {torch.max(torch.abs(out_torch - out_triton))}')
