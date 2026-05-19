import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024

@triton.jit
def add_kernel(
    x_ptr, 
    y_ptr, 
    output_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out_vals = x_vals + y_vals

    tl.store(output_ptr + offsets, out_vals, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.device.type != 'cuda':
        x = x.to('cuda')
    if y.device.type != 'cuda':
        y = y.to('cuda')

    assert x.numel() == y.numel(), "Tensors must have the same number of elements"
    n_elements = x.numel()

    output = torch.empty_like(x)

    grid = lambda META: ((n_elements + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'],)
    add_kernel[grid](
        x, 
        y, 
        output, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output
