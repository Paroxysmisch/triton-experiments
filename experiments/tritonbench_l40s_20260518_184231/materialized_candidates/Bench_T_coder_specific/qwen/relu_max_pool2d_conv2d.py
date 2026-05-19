import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    I, W, B, O,
    stride, padding, dilation, groups,
    N, C, IH, IW, KH, KW, OH, OW
):
    pid = tl.program_id(axis=0)
    num_programs = N * OH * OW
    # Each program will handle one pixel in the output
    if pid >= num_programs:
        return

    n = pid // (OH * OW)
    oh = pid % OH
    ow = (pid % (OH * OW)) // OW
    oc = pid % (OH * OW) % OW

    # Compute the indices for the input tensor
    ih_base = oh * stride - padding + dilation * (KH - 1) // 2
    iw_base = ow * stride - padding + dilation * (KW - 1) // 2

    # Initialize the output value
    o_val = 0.0

    # Perform the convolution
    for kh in range(KH):
        for kw in range(KW):
            ih = ih_base + kh * dilation
            iw = iw_base + kw * dilation
            if 0 <= ih < IH and 0 <= iw < IW:
                ic = oc * groups + kh * (KW // groups) + kw
                o_val += I[n, ic, ih, iw] * W[oc, ic, kh, kw]

    # Add bias if provided
    if B is not None:
        o_val += B[oc]

    # Write the output
    O[n, oc, oh, ow] = o_val

@triton.jit
def max_pool2d_kernel(
    I, O,
    kernel_size, stride, padding, dilation,
    ceil_mode,
    N, C, IH, IW, OH, OW
):
    pid = tl.program_id(axis=0)
    num_programs = N * OH * OW
    if pid >= num_programs:
        return

    n = pid // (OH * OW)
    oh = pid % OH
    ow = (pid % (OH * OW)) // OW
    oc = pid % (OH * OW) % OW

    # Compute the indices for the input tensor
    ih_base = oh * stride - padding + dilation * (kernel_size - 1) // 2
    iw_base = ow * stride - padding + dilation * (kernel_size - 1) // 2

    # Initialize the maximum value
    o_val = float('-inf')

    # Perform the max pooling
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            ih = ih_base + kh * dilation
            iw = iw_base + kw * dilation
            if 0 <= ih < IH and 0 <= iw < IW:
                o_val = max(o_val, I[n, oc, ih, iw])

    # Write the output
    O[n, oc, oh, ow] = o_val

@triton.jit
def relu_kernel(
    I, O,
    N, C, H, W
):
    pid = tl.program_id(axis=0)
    num_programs = N * H * W
    if pid >= num_programs:
        return

    n = pid // (H * W)
    h = pid % H
    w = (pid % (H * W)) // W
    c = pid % (H * W) % W

    # Apply ReLU
    o_val = max(0.0, I[n, c, h, w])

    # Write the output
    O[n, c, h, w] = o_val

@triton.jit
def relu_max_pool2d_conv2d_forward(
    input, weight, bias, conv_out, pool_out, relu_out,
    conv_stride, conv_padding, conv_dilation, conv_groups,
    pool_kernel_size, pool_stride, pool_padding, pool_dilation,
    pool_ceil_mode,
    N, C, IH, IW, OC, OH, OW
):
    # Convolution
    conv2d_kernel[input, weight, bias, conv_out,
                  conv_stride, conv_padding, conv_dilation, conv_groups,
                  N, C, IH, IW, OC, OH, OW]

    # Max Pooling
    max_pool2d_kernel[conv_out, pool_out,
                      pool_kernel_size, pool_stride, pool_padding, pool_dilation,
                      pool_ceil_mode,
                      N, OC, OH, OW, OH, OW]

    # ReLU
    relu_kernel[pool_out, relu_out,
                 N, OC, OH, OW]

# Wrapper function
def relu_max_pool2d_conv2d(input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False):
    # Determine output shapes
    if pool_stride is None:
        pool_stride = pool_kernel_size
    OH = ((IH + 2 * pool_padding - pool_dilation * (pool_kernel_size - 1) - 1) // pool_stride) + 1
    OW = ((IW + 2 * pool_padding - pool_dilation * (pool_kernel_size - 1) - 1) // pool_stride) + 1

    # Allocate memory for outputs
    conv_out = tl.zeros((N, OC, OH, OW), dtype=input.dtype)
    pool_out = tl.zeros((N, OC, OH, OW), dtype=input.dtype)
    relu_out = tl.zeros((N, OC, OH, OW), dtype=input.dtype)

    # Call the forward kernel
    relu_max_pool2d_conv2d_forward[input, weight, bias, conv_out, pool_out, relu_out,
                                    conv_stride, conv_padding, conv_dilation, conv_groups,
                                    pool_kernel_size, pool_stride, pool_padding, pool_dilation,
                                    pool_ceil_mode,
                                    N, C, IH, IW, OC, OH, OW]

    # Return the result
    return relu_out
