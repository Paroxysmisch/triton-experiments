import torch
import triton
import triton.language as tl

@triton.jit
def selu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    input_stride,
    output_stride,
    alpha,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_offsets = offsets * input_stride
    x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)

    zero = 0.0
    positive_part = tl.maximum(zero, x)
    exp_x = tl.exp(x)
    negative_part = alpha * (exp_x - 1)
    negative_part = tl.minimum(zero, negative_part)
    result = scale * (positive_part + negative_part)

    output_offsets = offsets * output_stride
    tl.store(output_ptr + output_offsets, result, mask=mask)

def selu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if not inplace:
        output = torch.empty_like(input)
    else:
        output = input

    input_1d = input.view(-1)
    output_1d = output.view(-1)
    assert input_1d.numel() == output_1d.numel(), "Input and output must have the same number of elements."

    n_elements = input_1d.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    selu_kernel[grid](
        input_1d, output_1d, n_elements,
        input_1d.stride(0), output_1d.stride(0),
        alpha=1.6732632423543772848170429916717,
        scale=1.0507009873554804934193349852946,
        BLOCK_SIZE=1024,
    )
    return output
