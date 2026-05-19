import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def elu_linear_kernel(input, weight, bias, alpha, output,
                      inplace: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Triton kernel for applying linear transformation followed by ELU
    row_idx = tl.program_id(0)
    output_row_start = row_idx * BLOCK_SIZE
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < weight.shape[1]

    # Load input row
    x = tl.load(input + row_idx * BLOCK_SIZE + col_offsets, mask=mask, other=0)

    # Apply linear transformation
    y = tl.dot(weight.to(x.dtype), x)

    # Apply bias
    if bias is not None:
        y += tl.load(bias).to(x.dtype)

    # Apply ELU activation
    y = tl.where(y > 0, y, alpha * (tl.exp(y) - 1))

    # Store result
    if inplace:
        tl.store(input + output_row_start + col_offsets, y, mask=mask)
    else:
        tl.store(output + output_row_start + col_offsets, y, mask=mask)

def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False) -> Tensor:
    # Wrapper function for applying linear transformation followed by ELU
    assert input.shape[-1] == weight.shape[1], "Incompatible dimensions"

    if bias is not None:
        assert weight.shape[0] == bias.shape[0], "Incompatible dimensions"

    if inplace:
        assert input.stride(-1) == 1, "Input tensor must be contiguous"

    output_shape = list(input.shape)
    output_shape[-1] = weight.shape[0]

    # Allocate output tensor
    if inplace:
        output = input
    else:
        output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    # Define grid based on input size
    grid = lambda meta: (triton.cdiv(input.numel(), meta['BLOCK_SIZE']), )

    # Launch Triton kernel
    elu_linear_kernel[grid](input, weight, bias, alpha, output, inplace)

    if not inplace:
        return output

    return input
