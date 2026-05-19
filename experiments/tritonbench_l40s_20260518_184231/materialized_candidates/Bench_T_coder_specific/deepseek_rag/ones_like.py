import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume

@triton.jit
def ones_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 1.0, mask=mask)

def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    if layout is None:
        layout = input.layout

    out = torch.empty(input.size(), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)
    N = volume(input.size())
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch.cuda.device(device):
        ones_kernel[grid_fn](out, N, BLOCK_SIZE=1024)
    return out
