import triton
import triton.language as tl

@triton.jit
def relu_batch_norm_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_shape, weight_shape, bias_shape, output_shape,
    stride, padding, dilation, groups, bn_running_mean_ptr, bn_running_var_ptr,
    bn_weight_ptr, bn_bias_ptr, momentum, eps, inplace
):
    # Get grid size
    H, W = triton.program_id(2), triton.program_id(1)
    N, C_in, _, _ = input_shape
    C_out, _, kH, kW = weight_shape
    
    # Calculate input indices
    n = H * W
    c_in = C_in // groups
    kh = (n % (kH * kW)) // kW
    kw = n % kW
    iH = H * stride + kh * dilation - padding
    iW = W * stride + kw * dilation - padding
    
    # Check if within bounds
    if iH < 0 or iH >= input_shape[2] or iW < 0 or iW >= input_shape[3]:
        return
    
    # Compute convolution
    acc = 0.0
    for g in range(groups):
        for oc in range(C_out // groups):
            for ic in range(c_in):
                ci = g * c_in + ic
                for kh_off in range(kH):
                    for kw_off in range(kW):
                        ih = iH + kh_off * dilation
                        iw = iW + kw_off * dilation
                        acc += input_ptr[n * C_in * iH * iW + ci * iH * iW + ih * iW + iw] * weight_ptr[g * C_out * C_in // groups * kH * kW + oc * C_in // groups * kH * kW + ic * kH * kW + kh_off * kW + kw_off]
    
    # Add bias if provided
    if bias_ptr is not None:
        acc += bias_ptr[oc]
    
    # Batch normalization
    mean = acc / (kH * kW)
    var = 0.0
    for g in range(groups):
        for oc in range(C_out // groups):
            for ic in range(c_in):
                ci = g * c_in + ic
                for kh_off in range(kH):
                    for kw_off in range(kW):
                        ih = iH + kh_off * dilation
                        iw = iW + kw_off * dilation
                        var += (input_ptr[n * C_in * iH * iW + ci * iH * iW + ih * iW + iw] - mean) ** 2
    var /= (kH * kW)
    normed = (acc - mean) / tl.sqrt(var + eps)
    
    # Scale and shift with batch normalization parameters
    if bn_weight_ptr is not None and bn_bias_ptr is not None:
        normed = normed * bn_weight_ptr[oc] + bn_bias_ptr[oc]
    
    # Apply ReLU
    output_ptr[n * C_out * iH * iW + oc] = max(0.0, normed)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=2, num_warps=8),
    ],
    key=['N', 'C_in', 'iH', 'iW']
)
def relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, running_mean=None, running_var=None, bn_weight=None, bn_bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False):
    input_shape = input.shape
    weight_shape = weight.shape
    bias_shape = bias.shape if bias is not None else (weight_shape[0],)
    output_shape = (input_shape[0], weight_shape[0], input_shape[2], input_shape[3])
    
    threads_per_block = (32, 32, 1)
    blocks_per_grid = ((output_shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
                       (output_shape[1] + threads_per_block[1] - 1) // threads_per_block[1],
                       (output_shape[2] * output_shape[3]) // threads_per_block[2])
    
    output = input.new_empty(output_shape)
    relu_batch_norm_conv2d_kernel[
        grid=blocks_per_grid,
        block=threads_per_block,
        num_warps=4
    ](
        input.data_ptr(), weight.data_ptr(), bias.data_ptr() if bias is not None else None, output.data_ptr(),
        input_shape, weight_shape, bias_shape, output_shape,
        stride, padding, dilation, groups, running_mean.data_ptr() if running_mean is not None else None, running_var.data_ptr() if running_var is not None else None,
        bn_weight.data_ptr() if bn_weight is not None else None, bn_bias.data_ptr() if bn_bias is not None else None, momentum, eps, inplace
    )
    
    return output
