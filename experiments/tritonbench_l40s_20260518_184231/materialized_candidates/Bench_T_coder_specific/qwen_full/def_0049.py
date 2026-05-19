import torch
import triton
import triton.language as tl

@triton.jit
def leaky_relu_fused_2d(
    input, weight, bias, stride, padding, dilation, groups, negative_slope
):
    # Adjust padding to be a tuple for each dimension
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Calculate output dimensions
    batch, in_c, in_h, in_w = input.shape
    out_c, _, kernel_h, kernel_w = weight.shape
    out_h = triton.cdiv(in_h + 2 * padding[0] - dilation[0] * (kernel_h - 1), stride)
    out_w = triton.cdiv(in_w + 2 * padding[1] - dilation[1] * (kernel_w - 1), stride)

    # Create a range tensor for output indices
    out_id = tl.program_id(0)
    out_jd = tl.program_id(1)
    out_kd = tl.program_id(2)
    out_i = out_id // out_w
    out_j = (out_id % out_w) * stride - padding[1]
    out_k = out_jd * stride - padding[0]
    out_d = out_kd

    # Compute input indices
    in_b = out_i * out_h * out_w + out_j * out_h + out_k + tl.arange(0, 16)
    in_b = in_b // out_h // out_w, in_b // out_h % out_w, in_b % out_h, in_b % out_w
    in_b = in_b[0], in_b[0], in_b[0], in_b[0], in_b[1], in_b[1], in_b[1], in_b[1], in_b[2], in_b[2], in_b[2], in_b[2], in_b[3], in_b[3], in_b[3], in_b[3]
    in_c = tl.arange(0, 16) // (kernel_h * kernel_w)
    in_c = in_c[0], in_c[1], in_c[2], in_c[3], in_c[0], in_c[1], in_c[2], in_c[3], in_c[0], in_c[1], in_c[2], in_c[3], in_c[0], in_c[1], in_c[2], in_c[3]
    in_h = tl.arange(0, 16) % (kernel_h * kernel_w) // kernel_w
    in_h = in_h[0], in_h[1], in_h[2], in_h[3], in_h[4], in_h[5], in_h[6], in_h[7], in_h[8], in_h[9], in_h[10], in_h[11], in_h[12], in_h[13], in_h[14], in_h[15]
    in_w = tl.arange(0, 16) % (kernel_h * kernel_w) % kernel_w
    in_w = in_w[0], in_w[1], in_w[2], in_w[3], in_w[4], in_w[5], in_w[6], in_w[7], in_w[8], in_w[9], in_w[10], in_w[11], in_w[12], in_w[13], in_w[14], in_w[15]

    # Compute weight indices
    wei_c = in_c // groups
    wei_k = in_h * kernel_w + in_w
    # Compute bias index
    bias_k = tl.arange(0, 16) // (kernel_h * kernel_w)
    # Load input, weight, and bias
    I = tl.load(input + in_b[0] * stride[0] * in_h * in_w + in_b[1] * stride[1] * in_h * in_w + in_b[2] * stride[2] * in_h * in_w + in_b[3] * stride[3] * in_h * in_w + in_h * in_w + in_w, mask=(in_b[0] < batch) & (in_b[1] < batch) & (in_b[2] < batch) & (in_b[3] < batch) & (in_h < in_h) & (in_w < in_w), other=0)
    W = tl.load(weight + out_d * out_c * kernel_h * kernel_w + wei_c * kernel_h * kernel_w + wei_k, mask=(out_d < out_c) & (wei_c < in_c) & (wei_k < kernel_h * kernel_w), other=0)
    B = tl.load(bias + out_d * 16 + bias_k, mask=(out_d < out_c) & (bias_k < 16), other=0)

    # Perform matrix multiplication and add bias
    O = tl.dot(I, W.T) + B

    # Apply Leaky ReLU activation
    O = tl.where(I > 0, O, negative_slope * O)

    # Store output
    tl.store(input + out_id * out_h * out_w + out_j * out_h + out_k + tl.arange(0, 16), O, mask=(out_j < out_w) & (out_k < out_h) & (tl.arange(0, 16) < out_h), eviction_policy='evict_first')

def leaky_relu_conv2d_triton(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False) -> torch.Tensor:
    # Ensure input is contiguous
    if not input.is_contiguous():
        input = input.contiguous()

    # Prepare bias for addition if provided
    if bias is not None:
        bias = bias.unsqueeze(0)

    # Adjust stride to be a tuple for each dimension
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Create a range tensor for output indices
    out_h = triton.cdiv(input.shape[2] + 2 * padding[0] - dilation[0] * (weight.shape[2] - 1), stride[0])
    out_w = triton.cdiv(input.shape[3] + 2 * padding[1] - dilation[1] * (weight.shape[3] - 1), stride[1])
    out_id = tl.arange(0, 16)
    out_jd = tl.arange(0, 16 // stride[0])
    out_kd = tl.arange(0, 16 // stride[1])
    out_i = out_id // out_w
    out_j = (out_id % out_w) * stride[0] - padding[1]
    out_k = out_jd * stride[1] - padding[0]
    out_d = out_kd

    # Apply the fused 2D convolution and Leaky ReLU kernel
    leaky_relu_fused_2d[(out_i, out_j, out_k, out_d)](input, weight, bias, stride, padding, dilation, groups, negative_slope)

    return input
