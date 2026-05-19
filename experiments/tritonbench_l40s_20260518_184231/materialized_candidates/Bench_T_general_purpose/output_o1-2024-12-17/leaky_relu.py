import triton
import triton.language as tl
import torch

@triton.jit
def _leaky_relu_kernel(inp_ptr, out_ptr, n_elements, negative_slope, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(inp_ptr + offsets, mask=mask)
    out = tl.where(x > 0, x, negative_slope * x)
    tl.store(out_ptr + offsets, out, mask=mask)

def leaky_relu(input, negative_slope=0.01, inplace=False):
    if not input.is_cuda:
        raise ValueError("Input must be a CUDA tensor.")
    if not input.is_contiguous():
        input = input.contiguous()

    output = input if inplace else torch.empty_like(input)
    n_elements = input.numel()

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _leaky_relu_kernel[grid](
        input.data_ptr(),
        output.data_ptr(),
        n_elements,
        negative_slope,
        BLOCK_SIZE=1024
    )
    return output
