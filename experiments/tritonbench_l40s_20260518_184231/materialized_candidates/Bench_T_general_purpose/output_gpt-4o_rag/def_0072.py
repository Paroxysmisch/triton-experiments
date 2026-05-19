import triton
import triton.language as tl

@triton.jit
def relu_batch_norm_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    stride, padding, dilation, groups,
    running_mean_ptr, running_var_ptr,
    bn_weight_ptr, bn_bias_ptr,
    training, momentum, eps,
    in_channels, out_channels, kernel_h, kernel_w,
    input_h, input_w, output_h, output_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Calculate batch, channel, and spatial indices
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    # Calculate the starting point for the input tensor
    in_channel_id = out_channel_id // groups
    group_id = out_channel_id % groups

    # Initialize accumulators for the convolution
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Iterate over the input channels
    for k in range(0, in_channels // groups, BLOCK_K):
        # Load input and weight tiles
        input_tile = tl.load(input_ptr + batch_id * in_channels * input_h * input_w + in_channel_id * input_h * input_w + h * stride * input_w + w * stride)
        weight_tile = tl.load(weight_ptr + out_channel_id * (in_channels // groups) * kernel_h * kernel_w + in_channel_id * kernel_h * kernel_w)

        # Perform convolution
        acc += tl.dot(input_tile, weight_tile)

    # Apply bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + out_channel_id)
        acc += bias

    # Batch normalization
    if training:
        mean = tl.mean(acc, axis=0)
        var = tl.var(acc, axis=0)
        if running_mean_ptr is not None and running_var_ptr is not None:
            running_mean = tl.load(running_mean_ptr + out_channel_id)
            running_var = tl.load(running_var_ptr + out_channel_id)
            running_mean = momentum * mean + (1 - momentum) * running_mean
            running_var = momentum * var + (1 - momentum) * running_var
            tl.store(running_mean_ptr + out_channel_id, running_mean)
            tl.store(running_var_ptr + out_channel_id, running_var)
    else:
        mean = tl.load(running_mean_ptr + out_channel_id)
        var = tl.load(running_var_ptr + out_channel_id)

    inv_std = 1.0 / tl.sqrt(var + eps)
    acc = (acc - mean) * inv_std

    # Apply batch normalization weights and biases
    if bn_weight_ptr is not None:
        bn_weight = tl.load(bn_weight_ptr + out_channel_id)
        acc *= bn_weight

    if bn_bias_ptr is not None:
        bn_bias = tl.load(bn_bias_ptr + out_channel_id)
        acc += bn_bias

    # Apply ReLU activation
    acc = tl.maximum(acc, 0)

    # Store the result
    tl.store(output_ptr + batch_id * out_channels * output_h * output_w + out_channel_id * output_h * output_w + h * output_w + w, acc)


def relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
                           running_mean=None, running_var=None, bn_weight=None, bn_bias=None,
                           training=False, momentum=0.1, eps=1e-5, inplace=False):
    # Input and output shapes
    minibatch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    # Output dimensions
    oH = (iH + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    oW = (iW + 2 * padding - dilation * (kW - 1) - 1) // stride + 1

    # Allocate output tensor
    output = torch.empty((minibatch, out_channels, oH, oW), device=input.device, dtype=input.dtype)

    # Launch Triton kernel
    grid = (minibatch, out_channels, oH, oW)
    relu_batch_norm_conv2d_kernel[grid](
        input, weight, bias, output,
        stride, padding, dilation, groups,
        running_mean, running_var,
        bn_weight, bn_bias,
        training, momentum, eps,
        in_channels, out_channels, kH, kW,
        iH, iW, oH, oW,
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=8
    )

    return output
