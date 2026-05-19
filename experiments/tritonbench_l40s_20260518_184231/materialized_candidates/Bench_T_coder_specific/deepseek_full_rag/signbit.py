import torch
import triton
import triton.language as tl

@triton.jit
def signbit_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_ptrs = input_ptr + offsets
    input = tl.load(input_ptrs, mask=mask, other=0.0)
    output = tl.signbit(input)
    output_ptrs = output_ptr + offsets
    tl.store(output_ptrs, output, mask=mask)


def signbit(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    signbit_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
