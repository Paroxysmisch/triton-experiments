import triton
import triton.language as tl
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union

@triton.jit
def gelu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_channels, out_channels, kH, kW,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    iH, iW, oH, oW, groups, approximate, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Define the position of the block
    pid = tl.program_id(axis=0)
    n = pid // (oH * oW)
    oh = (pid // oW) % oH
    ow = pid % oW

    # Initialize the output accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the input channels and perform convolution
    for c in range(in_channels // groups):
        for kh in range(kH):
            for kw in range(kW):
                ih = oh * stride_h - pad_h + kh * dilation_h
                iw = ow * stride_w - pad_w + kw * dilation_w
                if 0 <= ih < iH and 0 <= iw < iW:
                    input_offset = n * in_channels * iH * iW + c * iH * iW + ih * iW + iw
                    weight_offset = c * kH * kW + kh * kW + kw
                    acc += tl.load(input_ptr + input_offset) * tl.load(weight_ptr + weight_offset)

    # Add bias if provided
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + n)

    # Apply GELU activation
    if approximate == 'tanh':
        acc = 0.5 * acc * (1 + tl.tanh((tl.sqrt(2 / tl.pi) * (acc + 0.044715 * acc ** 3))))
    else:
        acc = acc * 0.5 * (1.0 + tl.erf(acc / tl.sqrt(2.0)))

    # Store the result
    output_offset = n * out_channels * oH * oW + oh * oW + ow
    tl.store(output_ptr + output_offset, acc)


def gelu_conv2d(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int], str] = 0,
    dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, approximate: str = 'none',
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # Parse stride, padding, and dilation
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Calculate output dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    oH = (iH + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    oW = (iW + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1

    # Prepare output tensor
    if out is None:
        out = torch.empty((batch_size, out_channels, oH, oW), device=input.device, dtype=input.dtype)

    # Launch Triton kernel
    grid = (batch_size * oH * oW,)
    gelu_conv2d_kernel[grid](
        input, weight, bias, out,
        in_channels, out_channels, kH, kW,
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        iH, iW, oH, oW, groups, approximate,
        BLOCK_M=32, BLOCK_N=32
    )

    return out
