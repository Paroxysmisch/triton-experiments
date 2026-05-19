import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def leaky_relu_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    N,
    C,
    H,
    W,
    KN,
    KH,
    KW,
    stride_n,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    negative_slope: tl.constexpr,
):
    """
    leaky_relu_conv2d kernel
    """
    # Map the program id to the batch element to process
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    # Compute the position of the output element
    out_offset = ((n * KN + c) * H + h) * W + w

    # Compute the position of the input elements
    in_offset = out_offset // (stride_h * stride_w)
    in_n = in_offset // (H * W)
    in_h = (in_offset % (H * W)) // W
    in_w = (in_offset % (H * W)) % W

    # Compute the position of the weight elements
    w_offset = (c * KH + in_h * dilation_h) * KW + in_w * dilation_w

    # Initialize accumulator
    acc = tl.zeros([KN, KH, KW], dtype=tl.float32)

    # Loop over the kernel size
    for i in range(0, KN):
        for j in range(0, KH):
            for k in range(0, KW):
                # Compute the position of the input element
                in_h = h + (j - pad_h) * dilation_h
                in_w = w + (k - pad_w) * dilation_w
                in_offset = ((in_n * N + (i + c * KN) % N) * H + in_h) * W + in_w
                in_mask = (in_h < H) & (in_w < W)

                # Load the input and weight elements
                in_val = tl.load(input_ptr + in_offset, mask=in_mask)
                w_val = tl.load(weight_ptr + w_offset)

                # Accumulate the result
                acc += in_val * w_val

    # Add bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + c)
        acc += bias_val

    # Apply leaky relu activation
    acc = tl.where(acc >= 0, acc, negative_slope * acc)

    # Store the result
    tl.store(output_ptr + out_offset, acc)

def leaky_relu_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    negative_slope: float = 0.01,
    inplace: bool = False,
) -> Tensor:
    """
    Applies a 2D convolution operator over an input signal composed of several input planes,
    followed by applying the Leaky ReLU activation function.
    """
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)

    kernel_size = weight.shape[-2:]
    assert len(kernel_size) == 2, "Kernel size must be a tuple of two values"
    assert weight.shape[1] % groups == 0, "Number of input channels must be divisible by number of groups"
    assert weight.shape[0] % groups == 0, "Number of output channels must be divisible by number of groups"

    out_channels, in_channels_per_group = weight.shape[0], weight.shape[1] // groups
    _, in_h, in_w = input.shape
    out_h, out_w = (in_h + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1), in_w + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1))

    if inplace:
        assert input.shape[1:] == (in_channels_per_group, in_h, in_w), "Input shape must match in_channels_per_group, in_h, in_w"
        assert weight.shape == (out_channels, in_channels_per_group), "Weight shape must be (out_channels, in_channels_per_group)"
        assert bias is None or bias.shape == (out_channels,), "Bias shape must be (out_channels,)"
        assert not input.requires_grad and not weight.requires_grad and (bias is None or not bias.requires_grad), "Inplace operation must be performed on non-volatile tensors"

        output = input
    else:
        assert input.shape[1:] == (in_channels_per_group, in_h, in_w), "Input shape must match in_channels_per_group, in_h, in_w"
        assert weight.shape == (out_channels, in_channels_per_group), "Weight shape must be (out_channels, in_channels_per_group)"
        assert bias is None or bias.shape == (out_channels,), "Bias shape must be (out_channels,)"

        output = torch.empty((input.shape[0], out_channels, out_h, out_w), device=input.device, dtype=input.dtype)

    num_warps = 4
    grid = lambda meta: (
        input.shape[0],
        out_channels // meta["GROUP_SIZE"],
        out_h,
        out_w,
    )

    leaky_relu_conv2d_kernel[grid](
        input,
        weight,
        output,
        bias,
        input.shape[0],
        in_channels_per_group,
        in_h,
        in_w,
        out_channels,
        kernel_size[0],
        kernel_size[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        negative_slope,
        num_warps=num_warps,
    )

    return output
