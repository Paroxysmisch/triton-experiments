import triton
import triton.language as tl
import torch

@triton.jit
def exp_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.exp(input)
    tl.store(output_ptr + offsets, output, mask=mask)

def exp(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor shape mismatch"
        assert out.dtype == input.dtype, "Output tensor dtype mismatch"
        assert out.device == input.device, "Output tensor device mismatch"
    n_elements = input.numel()
    if n_elements == 0:
        return out
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    exp_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
