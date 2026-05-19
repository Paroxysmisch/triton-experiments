import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    inplace: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.maximum(input, 0.0)
    if inplace:
        tl.store(input_ptr + offsets, output, mask=mask)
    else:
        tl.store(output_ptr + offsets, output, mask=mask)

def relu_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int or tuple = 1,
    padding: int or tuple or str = 0,
    dilation: int or tuple = 1,
    groups: int = 1,
    inplace: bool = False
) -> torch.Tensor:
    # Perform the 2D convolution using PyTorch's function
    x = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    # Apply ReLU using the Triton kernel
    if inplace:
        if not x.is_contiguous():
            x = x.contiguous()
        grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
        relu_kernel[grid](x, x, x.numel(), inplace=True, BLOCK_SIZE=1024)
        return x
    else:
        output = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
        relu_kernel[grid](output, x, x.numel(), inplace=False, BLOCK_SIZE=1024)
        return output
