import triton
import torch

# Triton kernel to compute the exponential and square root
@triton.jit
def exp_sqrt_kernel(input, out, n_elements, BLOCK_SIZE: tl.constexpr(1024)):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    inp = tl.load(input + offset, mask=mask)
    out = tl.sqrt(tl.exp(inp))
    tl.store(out + offset, out, mask=mask)

# Wrapper function to call the Triton kernel
def exp_sqrt(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    exp_sqrt_kernel[grid](input, out, n_elements)
    return out
