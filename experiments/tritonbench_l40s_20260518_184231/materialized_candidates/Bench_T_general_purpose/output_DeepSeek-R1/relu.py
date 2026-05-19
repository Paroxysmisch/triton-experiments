import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(input_ptr + offsets, mask=mask)
    x_relu = tl.where(x > 0, x, 0)
    tl.store(output_ptr + offsets, x_relu, mask=mask)

def relu(input, inplace=False):
    if not inplace:
        output = torch.empty_like(input)
    else:
        output = input
    numel = input.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](input, output, numel, BLOCK_SIZE=1024)
    return output
