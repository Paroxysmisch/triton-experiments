import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr, bias_ptr, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,
    N, C, IH, IW, K, KH, KW, OC, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
):
    """
    Applies a 2D convolution over an input image composed of several input planes.

    Parameters:
    -----------
    input_ptr : tl.tensor
        Pointer to the input tensor in global memory.
    weight_ptr : tl.tensor
        Pointer to the weight tensor in global memory.
    output_ptr : tl.tensor
        Pointer to the output tensor where the result will be stored.
    bias_ptr : tl.tensor
        Pointer to the bias tensor in global memory.
    stride_h : int
        Stride of the convolution kernel in height.
    stride_w : int
        Stride of the convolution kernel in width.
    padding_h : int
        Padding of the input in height.
    padding_w : int
        Padding of the input in width.
    dilation_h : int
        Dilation of the convolution kernel in height.
    dilation_w : int
        Dilation of the convolution kernel in width.
    groups : int
        Number of groups for grouped convolution.
    N : int
        Batch size.
    C : int
        Number of input channels.
    IH : int
        Height of the input image.
    IW : int
        Width of the input image.
    K : int
        Number of output channels.
    KH : int
        Height of the convolution kernel.
    KW : int
        Width of the convolution kernel.
    OC : int
        Number of output channels.
    BLOCK_SIZE_H : int
        Block size in height.
    BLOCK_SIZE_W : int
        Block size in width.
    BLOCK_SIZE_C : int
        Block size in channel.
    """

    # Get the program ID for each dimension
    n = tl.program_id(axis=0)
    c = tl.program_id(axis=1)
    kh = tl.program_id(axis=2)
    kw = tl.program_id(axis=3)

    # Compute indices for the current element
    nx = tl.program_id(axis=4)
    ny = tl.program_id(axis=5)

    # Compute base index for the current element
    input_base = n * C * (IH + 2 * padding_h) * (IW + 2 * padding_w) + c * (IH + 2 * padding_h) * (IW + 2 * padding_w)
    weight_base = c * KH * KW
    output_base = n * OC * IH * IW + c * IH * IW

    # Compute effective input indices considering padding and dilation
    ix = nx * stride_w + kw * dilation_w - padding_w
    iy = ny * stride_h + kh * dilation_h - padding_h

    # Initialize the output value
    acc = tl.zeros([], dtype=tl.float32)

    # Iterate over the receptive field
    for ic in range(C // groups):
        for ph in range(KH):
            for pw in range(KW):
                input_idx = input_base + ic * (IH + 2 * padding_h) * (IW + 2 * padding_w) + (iy + ph * dilation_h) * (IW + 2 * padding_w) + (ix + pw * dilation_w)
                weight_idx = weight_base + ic * KH * KW + ph * KW + pw
                acc += tl.load(input_ptr + input_idx) * tl.load(weight_ptr + weight_idx)

    # Add bias if provided
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + c)

    # Store the result
    output_idx = output_base + ny * IW + nx
    tl.store(output_ptr + output_idx, acc)
