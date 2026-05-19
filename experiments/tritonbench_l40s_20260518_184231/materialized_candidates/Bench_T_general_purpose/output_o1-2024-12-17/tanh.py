import triton
import triton.language as tl
import torch


@triton.jit
def _tanh_kernel(
    in_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    y = 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0
    tl.store(out_ptr + offsets, y, mask=mask)


def tanh(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    assert input.is_cuda, "input must be a CUDA tensor"
    assert out.is_cuda, "out must be a CUDA tensor"

    n_elements = input.numel()
    grid = ( (n_elements + 1023) // 1024, )
    _tanh_kernel[grid](input.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE=1024)
    return out
