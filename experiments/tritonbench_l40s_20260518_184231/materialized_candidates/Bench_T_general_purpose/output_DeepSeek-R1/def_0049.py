import torch
import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(
    input_ptr,
    output_ptr,
    negative_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    zero = tl.zeros_like(x)
    positive_part = tl.maximum(zero, x)
    negative_part = tl.minimum(zero, x) * negative_slope
    output = positive_part + negative_part
    tl.store(output_ptr + offsets, output, mask=mask)

def leaky_relu_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    negative_slope: float = 0.01,
    inplace: bool = False,
) -> torch.Tensor:
    # Perform the 2D convolution
    conv_output = torch.nn.functional.conv2d(
        input, weight, bias, stride, padding, dilation, groups
    )
    
    # Determine if inplace operation is possible
    if inplace:
        output = conv_output
    else:
        output = torch.empty_like(conv_output)
    
    # Compute total number of elements
    n_elements = conv_output.numel()
    
    # Grid launch heuristic
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the Leaky ReLU kernel
    leaky_relu_kernel[grid](
        conv_output, output, negative_slope, n_elements, BLOCK_SIZE=1024
    )
    
    return output
