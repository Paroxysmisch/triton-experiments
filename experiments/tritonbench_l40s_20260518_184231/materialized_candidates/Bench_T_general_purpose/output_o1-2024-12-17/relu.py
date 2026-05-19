import triton
import triton.language as tl
import torch

@triton.jit
def _relu_kernel(
    in_ptr, 
    out_ptr,
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)

def relu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")

    n_elements = input.numel()

    if inplace:
        out = input
    else:
        out = torch.empty_like(input)

    grid = lambda meta: ( (n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    
    _relu_kernel[grid](
        input.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE=1024
    )

    return out
