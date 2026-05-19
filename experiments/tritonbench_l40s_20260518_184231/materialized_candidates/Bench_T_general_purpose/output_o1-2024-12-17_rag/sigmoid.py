import torch
import triton
import triton.language as tl

@triton.jit
def _sigmoid_kernel(
    x_ptr,          # *Pointer* to the input tensor
    out_ptr,        # *Pointer* to the output tensor
    n_elements,     # Number of elements to process
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    result = 1.0 / (1.0 + tl.exp(-x))

    tl.store(out_ptr + offsets, result, mask=mask)

def sigmoid(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    assert input.is_cuda and out.is_cuda

    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _sigmoid_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)

    return out
