def fused_instance_norm_selu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_features=None, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
    # Convolution
    output = triton.ops.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    # SELU activation
    output = triton.ops.selu(output)
    
    # Instance normalization
    output = triton.ops.instance_norm(output, num_features, eps, momentum, affine, track_running_stats)
    
    return output
