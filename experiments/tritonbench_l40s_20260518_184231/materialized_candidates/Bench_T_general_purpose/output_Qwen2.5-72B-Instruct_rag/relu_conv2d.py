import torch
import triton
import triton.language as tl

@triton.jit
def relu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_shape, weight_shape, output_shape,
    stride, padding, dilation, groups,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    """
    Applies a 2D convolution followed by ReLU activation.

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
    input_shape : tuple
        Shape of the input tensor (N, C, H, W).
    weight_shape : tuple
        Shape of the weight tensor (O, I, K, K).
    output_shape : tuple
        Shape of the output tensor (N, O, H_out, W_out).
    stride : int or tuple
        Stride of the convolution.
    padding : int or tuple
        Padding added to all sides of the input.
    dilation : int or tuple
        Spacing between kernel elements.
    groups : int
        Number of blocked connections from input channels to output channels.
    BLOCK_SIZE_N : tl.constexpr
        Block size for the batch dimension.
    BLOCK_SIZE_H : tl.constexpr
        Block size for the height dimension.
    BLOCK_SIZE_W : tl.constexpr
        Block size for the width dimension.
    """
    n, c, h, w = input_shape
    o, ic, kh, kw = weight_shape
    _, _, h_out, w_out = output_shape

    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    n_start = pid_n * BLOCK_SIZE_N
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W

    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)

    mask_n = n_offsets < n
    mask_h = h_offsets < h_out
    mask_w = w_offsets < w_out

    output_offsets = n_offsets[:, None, None] * o * h_out * w_out + \
                     h_offsets[None, :, None] * w_out + \
                     w_offsets[None, None, :]

    output = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    for oc in range(o):
        for icg in range(ic // groups):
            for kh in range(kh):
                for kw in range(kw):
                    h_in = h_offsets * stride - padding + kh * dilation
                    w_in = w_offsets * stride - padding + kw * dilation
                    mask_h_in = (h_in >= 0) & (h_in < h)
                    mask_w_in = (w_in >= 0) & (w_in < w)
                    mask_in = mask_h_in & mask_w_in & mask_n

                    input_idx = n_offsets[:, None, None] * c * h * w + \
                                (icg * groups + pid_n) * h * w + \
                                h_in[None, :, None] * w + \
                                w_in[None, None, :]

                    weight_idx = oc * ic * kh * kw + icg * kh * kw + kh * kw + kw

                    input_val = tl.load(input_ptr + input_idx, mask=mask_in, other=0.0)
                    weight_val = tl.load(weight_ptr + weight_idx)

                    output += input_val * weight_val

        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + oc)
            output += bias_val

    output = tl.maximum(output, 0.0)  # Apply ReLU

    tl.store(output_ptr + output_offsets, output, mask=mask_n & mask_h & mask_w)

def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    """
    Applies a 2D convolution followed by ReLU activation.

    Parameters:
    -----------
    input : torch.Tensor
        The input tensor of shape (minibatch, in_channels, iH, iW).
    weight : torch.Tensor
        The convolution filters of shape (out_channels, in_channels / groups, kH, kW).
    bias : torch.Tensor, optional
        Optional bias tensor of shape (out_channels). Default: None.
    stride : int or tuple, optional
        The stride of the convolution kernel. Default: 1.
    padding : int, tuple, or string, optional
        Padding added to all sides of the input. Default: 0.
    dilation : int or tuple, optional
        The spacing between kernel elements. Default: 1.
    groups : int, optional
        Number of blocked connections from input channels to output channels. Default: 1.
    inplace : bool, optional
        If True, will perform ReLU operation in-place. Default: False.

    Returns:
    --------
    torch.Tensor
        A tensor resulting from the 2D convolution followed by ReLU activation.
    """
    device = input.device
    assert device.type == "cuda", "Input tensor must be on a CUDA device for triton ops."

    N, C, H, W = input.shape
    O, IC, KH, KW = weight.shape
    IC = IC * groups
    assert C % groups == 0, "Input channels must be divisible by groups"

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    H_out = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) // stride[0] + 1
    W_out = (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) // stride[1] + 1

    output_shape = (N, O, H_out, W_out)
    output = torch.empty(output_shape, device=device)

    BLOCK_SIZE_N = 16
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    grid = (N // BLOCK_SIZE_N, H_out // BLOCK_SIZE_H, W_out // BLOCK_SIZE_W)

    relu_conv2d_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        input_shape=(N, C, H, W),
        weight_shape=(O, IC // groups, KH, KW),
        output_shape=output_shape,
        stride=stride[0],
        padding=padding[0],
        dilation=dilation[0],
        groups=groups,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )

    if inplace:
        input.copy_(output)
        return input
    else:
        return output
