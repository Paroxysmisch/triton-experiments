import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume

# Triton kernel for division
@triton.jit
def div_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x / y
    tl.store(output_ptr + offsets, output, mask=mask)

# Wrapper function for calling the Triton kernel
def div(x: torch.Tensor, y: torch.Tensor, *, rounding_mode=None, out=None):
    assert x.is_cuda and y.is_cuda
    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.is_cuda
    n_elements = volume(x.shape)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    div_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out
