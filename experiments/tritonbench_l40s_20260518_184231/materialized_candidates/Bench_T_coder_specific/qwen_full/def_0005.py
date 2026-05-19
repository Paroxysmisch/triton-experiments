import torch
import triton
import triton.language as tl

@triton.jit
def relu_sqrt_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = tl.where(x >= 0, x, 0)
    x = tl.sqrt(x)
    tl.store(y_ptr + offsets, x, mask=mask)

def relu_sqrt(input, inplace=False, out=None) -> torch.Tensor:
    assert input.is_contiguous()
    assert input.dim() == 1

    if inplace:
        assert out is None
        relu_sqrt_kernel[(input.numel(),)](input, input, input.numel(), 1024)
        return input
    else:
        if out is None:
            output = torch.empty_like(input)
        else:
            output = out
        relu_sqrt_kernel[(input.numel(),)](input, output, input.numel(), 1024)
        return output
