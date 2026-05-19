import triton
import triton.language as tl
import torch

@triton.jit
def _selu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    x_stride,
    output_stride,
    alpha,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets * x_stride, mask=mask, other=0.0)
    # Compute SELU components
    positive = tl.maximum(x, 0.0)
    exp_x = tl.exp(x)
    negative = tl.minimum(alpha * (exp_x - 1.0), 0.0)
    selu_val = scale * (positive + negative)
    tl.store(output_ptr + offsets * output_stride, selu_val, mask=mask)

def selu(input, inplace=False):
    assert input.is_cuda, "Input tensor must be on CUDA device"
    if not inplace:
        output = torch.empty_like(input)
    else:
        output = input
    n_elements = input.numel()
    if n_elements == 0:
        return output  # Handle empty tensor
    # Determine strides for the last dimension
    x_stride = input.stride(-1) if input.ndim > 0 else 1
    output_stride = output.stride(-1) if output.ndim > 0 else 1
    # Launch kernel with appropriate grid and block size
    BLOCK_SIZE = 1024  # Can be tuned for optimal performance
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    alpha_val = 1.6732632423543772848170429916717
    scale_val = 1.0507009873554804934193349852946
    _selu_kernel[grid](input, output, n_elements, x_stride, output_stride, alpha_val, scale_val, BLOCK_SIZE=BLOCK_SIZE)
    return output
