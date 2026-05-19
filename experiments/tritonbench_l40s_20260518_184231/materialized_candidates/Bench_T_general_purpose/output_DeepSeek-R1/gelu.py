import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    input_row_stride,
    output_row_stride,
    APPROXIMATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate input and output offsets based on strides
    input_offsets = offsets * input_row_stride
    x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)

    if APPROXIMATE == 0:
        # Exact GELU using erf
        cdf = 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865475))  # 1 / sqrt(2)
        output = x * cdf
    else:
        # Approximate GELU using tanh
        a = tl.math.sqrt(2.0 / 3.141592653589793)
        x_cubed = x * x * x
        inner = a * (x + 0.044715 * x_cubed)
        tanh_inner = tl.math.tanh(inner)
        output = 0.5 * x * (1.0 + tanh_inner)

    # Store the result
    output_offsets = offsets * output_row_stride
    tl.store(output_ptr + output_offsets, output, mask=mask)

def gelu(input, approximate='none'):
    output = torch.empty_like(input)
    n_elements = input.numel()
    if n_elements == 0:
        return output
    # Calculate strides for the last dimension
    input_row_stride = input.stride(-1) if input.dim() > 0 else 1
    output_row_stride = output.stride(-1) if output.dim() > 0 else 1
    # Determine the approximate mode
    approximate_mode = 0 if approximate == 'none' else 1
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](
        input, output, n_elements,
        input_row_stride, output_row_stride,
        approximate_mode,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output
