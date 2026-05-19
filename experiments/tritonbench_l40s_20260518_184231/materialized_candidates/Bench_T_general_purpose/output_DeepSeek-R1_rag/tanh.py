import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def tanh_kernel(
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
    output = tl.tanh(input)
    tl.store(output_ptr + offsets, output, mask=mask)

def tanh(input: torch.Tensor, *, out=None):
    assert input.is_cuda, "Input tensor must be on GPU"
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_cuda, "Output tensor must be on GPU"
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    tanh_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
