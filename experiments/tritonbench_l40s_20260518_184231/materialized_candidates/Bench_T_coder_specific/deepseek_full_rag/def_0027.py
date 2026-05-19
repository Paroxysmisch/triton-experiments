import triton
import torch

@triton.jit
def sqrt_tanh_kernel(input, out, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate offset
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create mask
    mask = offset < n_elements
    # Load input
    x = tl.load(input + offset, mask=mask)
    # Apply sqrt
    x = tl.sqrt(x)
    # Apply tanh
    x = tl.math.tanh(x)
    # Store output
    tl.store(out + offset, x, mask=mask)

def sqrt_tanh(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sqrt_tanh_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
