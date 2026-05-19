import torch
import triton
import triton.language as tl

# Triton kernel for 2D convolution followed by sigmoid activation
@triton.jit
def sigmoid_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    in_channels, out_channels, in_h, in_w, kernel_h, kernel_w, groups,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
):
    """
    Applies a 2D convolution followed by the sigmoid activation function.

    Parameters:
    -----------
    input_ptr : tl.tensor
        Pointer to the input tensor in global memory.
    weight_ptr : tl.tensor
        Pointer to the weight tensor in global memory.
    bias_ptr : tl.tensor
        Pointer to the bias tensor in global memory.
    output_ptr : tl.tensor
        Pointer to the output tensor where the result will be stored.
    stride_h, stride_w : int
        Stride of the convolution kernel in height and width.
    padding_h, padding_w : int
        Padding on both sides of the input in height and width.
    dilation_h, dilation_w : int
        Dilation of the convolution kernel in height and width.
    in_channels, out_channels : int
        Number of input and output channels.
    in_h, in_w : int
        Height and width of the input tensor.
    kernel_h, kernel_w : int
        Height and width of the convolution kernel.
    groups : int
        Number of groups to split the input into.
    BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C : tl.constexpr
        Block sizes for the Triton grid in height, width, and channels.
    """
    batch_id = tl.program_id(0)
    block_h = tl.program_id(1) * BLOCK_SIZE_H
    block_w = tl.program_id(2) * BLOCK_SIZE_W

    offsets_h = block_h + tl.arange(0, BLOCK_SIZE_H)
    offsets_w = block_w + tl.arange(0, BLOCK_SIZE_W)
    offsets_c = tl.arange(0, BLOCK_SIZE_C)

    mask_h = (offsets_h < (in_h + 2 * padding_h - (kernel_h - 1) * dilation_h) // stride_h)
    mask_w = (offsets_w < (in_w + 2 * padding_w - (kernel_w - 1) * dilation_w) // stride_w)
    mask_c = (offsets_c < in_channels)

    result = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), dtype=tl.float32)

    for k_h in range(kernel_h):
        for k_w in range(kernel_w):
            input_h = (offsets_h * stride_h - padding_h + k_h * dilation_h)
            input_w = (offsets_w * stride_w - padding_w + k_w * dilation_w)
            input_mask_h = (input_h >= 0) & (input_h < in_h)
            input_mask_w = (input_w >= 0) & (input_w < in_w)
            input_mask = input_mask_h & input_mask_w & mask_h & mask_w & mask_c

            input_val = tl.load(
                input_ptr + batch_id * in_channels * in_h * in_w + offsets_c * in_h * in_w + input_h * in_w + input_w,
                mask=input_mask, other=0.0
            )

            weight_val = tl.load(
                weight_ptr + offsets_c * kernel_h * kernel_w + k_h * kernel_w + k_w,
                mask=mask_c, other=0.0
            )

            result += input_val * weight_val

    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + offsets_c, mask=mask_c, other=0.0)
        result += bias_val

    result = 1 / (1 + tl.exp(-result))

    output_mask = mask_h[:, None] & mask_w[None, :] & mask_c[None, None, :]
    tl.store(
        output_ptr + batch_id * out_channels * (in_h + 2 * padding_h - (kernel_h - 1) * dilation_h) // stride_h * (in_w + 2 * padding_w - (kernel_w - 1) * dilation_w) // stride_w + offsets_h * (in_w + 2 * padding_w - (kernel_w - 1) * dilation_w) // stride_w + offsets_w,
        result, mask=output_mask
    )

# Wrapper function for the Triton kernel
def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    """
    Applies a 2D convolution followed by the sigmoid activation function.

    Parameters:
    -----------
    input : torch.Tensor
        The input tensor of shape `(minibatch, in_channels, iH, iW)`.
    weight : torch.Tensor
        The convolution filters of shape `(out_channels, in_channels / groups, kH, kW)`.
    bias : torch.Tensor, optional
        Optional bias tensor of shape `(out_channels)`. Default: None.
    stride : int or tuple, optional
        The stride of the convolution kernel. Can be a single number or a tuple `(sH, sW)`. Default: 1.
    padding : int, tuple, or string, optional
        Padding on both sides of the input. Can be 'valid', 'same', single number, or tuple `(padH, padW)`. Default: 0.
    dilation : int or tuple, optional
        The spacing between kernel elements. Default: 1.
    groups : int, optional
        Number of groups to split the input into. Default: 1.
    out : torch.Tensor, optional
        The output tensor.

    Returns:
    --------
    torch.Tensor
        The output tensor of shape `(minibatch, out_channels, oH, oW)`.

    Math:
    \text{out} = \sigma(\text{conv2d}(\text{input}, \text{weight})) where \sigma(x) = \frac{1}{1 + e^{-x}} is the sigmoid function.
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    minibatch, in_channels, in_h, in_w = input.shape
    out_channels, in_channels_per_group, kernel_h, kernel_w = weight.shape

    if bias is not None:
        assert bias.shape == (out_channels,), "Bias shape must be (out_channels,)"

    out_h = (in_h + 2 * padding[0] - (kernel_h - 1) * dilation[0]) // stride[0]
    out_w = (in_w + 2 * padding[1] - (kernel_w - 1) * dilation[1]) // stride[1]

    if out is None:
        out = torch.empty((minibatch, out_channels, out_h, out_w), device=input.device, dtype=input.dtype)

    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 16

    grid = (minibatch, (out_h + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H, (out_w + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)

    sigmoid_conv2d_kernel[grid](
        input, weight, bias, out,
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        in_channels, out_channels, in_h, in_w, kernel_h, kernel_w, groups,
        BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
    )

    return out
