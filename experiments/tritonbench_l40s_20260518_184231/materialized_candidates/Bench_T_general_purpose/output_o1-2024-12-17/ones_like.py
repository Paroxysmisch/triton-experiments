import torch
import triton
import triton.language as tl

@triton.jit
def _fill_ones_kernel(data_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(data_ptr + offsets, 1.0, mask=mask)

def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    if dtype is None:
        dtype = input.dtype
    if layout is None:
        layout = input.layout
    if device is None:
        device = input.device

    out = torch.empty_like(
        input, 
        dtype=dtype, 
        layout=layout, 
        device=device, 
        requires_grad=requires_grad, 
        memory_format=memory_format
    )

    n_elements = out.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    _fill_ones_kernel[grid](out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out
