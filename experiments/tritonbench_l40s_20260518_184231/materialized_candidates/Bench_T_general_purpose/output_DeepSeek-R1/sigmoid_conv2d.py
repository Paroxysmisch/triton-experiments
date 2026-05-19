import torch
import triton
import triton.language as tl
from torch.nn import functional as F

@triton.jit
def sigmoid_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    output = 1.0 / (1.0 + tl.exp(-x))
    tl.store(output_ptr + offsets, output, mask=mask)

def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    # Process padding if it's a string
    if isinstance(padding, str):
        if padding.lower() == 'valid':
            pad = 0
        elif padding.lower() == 'same':
            kH, kW = weight.shape[2], weight.shape[3]
            stride_h = stride if isinstance(stride, int) else stride[0]
            stride_w = stride if isinstance(stride, int) else stride[1] if len(stride) > 1 else stride_h
            dilation_h = dilation if isinstance(dilation, int) else dilation[0]
            dilation_w = dilation if isinstance(dilation, int) else dilation[1] if len(dilation) > 1 else dilation_h
            effective_kH = (kH - 1) * dilation_h + 1
            effective_kW = (kW - 1) * dilation_w + 1
            pad_h = ((input.shape[2] - 1) * stride_h + effective_kH - input.shape[2]) // 2
            pad_w = ((input.shape[3] - 1) * stride_w + effective_kW - input.shape[3]) // 2
            pad = (pad_h, pad_w)
        else:
            raise ValueError("padding must be 'valid', 'same', or a tuple/int")
    else:
        pad = padding

    # Compute convolution
    conv_output = F.conv2d(input, weight, bias, stride, pad, dilation, groups)
    
    # Apply sigmoid using Triton kernel
    if out is None:
        out = torch.empty_like(conv_output)
    n_elements = conv_output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, 1024),)
    sigmoid_kernel[grid](out, conv_output, n_elements, BLOCK_SIZE=1024)
    return out
