import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    x_ptr, 
    y_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = tl.where(x > 0, x, 0.)
    tl.store(y_ptr + offsets, x, mask=mask)

def relu(input, inplace=False):
    x = input
    # Flatten the input for kernel launch
    x_flat = x.view(-1)
    n_elements = x_flat.numel()

    # Allocate or reuse output
    if inplace:
        y = x_flat
    else:
        y = torch.empty_like(x_flat)

    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    relu_kernel[grid](
        x_flat, 
        y, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return x if inplace else y.view_as(input)
