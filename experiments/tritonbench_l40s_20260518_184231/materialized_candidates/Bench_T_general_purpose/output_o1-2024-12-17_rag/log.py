import torch
import triton
import triton.language as tl

@triton.jit
def log_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.log(x)
    tl.store(output_ptr + offsets, y, mask=mask)

def log(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    log_kernel[grid](
        input, 
        out, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
