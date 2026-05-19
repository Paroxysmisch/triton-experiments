import torch
import triton
import triton.language as tl
from .utils import calculate_settings

@triton.jit
def dropout_relu_batch_norm_conv2d(
    input, weight, bias, stride, padding, dilation, groups, p, training, inplace
):
    output = conv2d(
        input, weight, bias, stride, padding, dilation, groups, inplace=inplace
    )
    output = batch_norm(output, training=training)
    output = relu(output)
    output = dropout(output, p, training)
    return output

def wrapper_dropout_relu_batch_norm_conv2d(
    input, weight, bias, stride, padding, dilation, groups, p, training, inplace
):
    assert (
        input.shape[1] % groups == 0
    ), "Number of input channels must be divisible by the number of groups"
    batch_norm_settings = calculate_settings(input)
    return dropout_relu_batch_norm_conv2d(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        p,
        training,
        inplace,
        **batch_norm_settings,
    )
