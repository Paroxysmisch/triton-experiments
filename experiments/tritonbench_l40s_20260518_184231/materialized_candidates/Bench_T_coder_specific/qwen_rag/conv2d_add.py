import triton
import triton.language as tl

@triton.jit
def conv2d_forward(
    input_ptr, weight_ptr, output_ptr, bias_ptr, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups, N, C, H, W, K, KH, KW
):
    """
    Perform a 2D convolution on the input tensor using the given weights and bias.
    
    Parameters:
    - input_ptr: Pointer to the input tensor in global memory.
    - weight_ptr: Pointer to the weight tensor in global memory.
    - output_ptr: Pointer to the output tensor where the result will be stored.
    - bias_ptr: Pointer to the bias tensor in global memory.
    - stride_h, stride_w: Stride values for height and width.
    - padding_h, padding_w: Padding values for height and width.
    - dilation_h, dilation_w: Dilation values for height and width.
    - groups: Number of groups for grouped convolution.
    - N: Batch size.
    - C: Number of input channels.
    - H: Height of the input tensor.
    - W: Width of the input tensor.
    - K: Number of output channels.
    - KH: Height of the convolution kernel.
    - KW: Width of the convolution kernel.
    """
    n = tl.program_id(0)
    c = tl.program_id(1)
    kh = tl.program_id(2)
    kw = tl.program_id(3)

    # Compute the effective spatial dimensions after padding
    padded_h = H + 2 * padding_h
    padded_w = W + 2 * padding_w

    # Compute the spatial indices within the padded tensor
    h_out = n * stride_h + kh * dilation_h
    w_out = c * stride_w + kw * dilation_w

    # Clamp the spatial indices to ensure they are within bounds
    h_in = max(0, min(h_out, padded_h))
    w_in = max(0, min(w_out, padded_w))

    # Adjust the indices to account for padding
    h_in -= padding_h
    w_in -= padding_w

    # Compute the linear index within the input tensor
    in_idx = n * C * padded_h * padded_w + c * padded_h * padded_w + h_in * padded_w + w_in

    # Load the input value
    input_val = tl.load(input_ptr + in_idx)

    # Initialize the accumulation variable
    accum = tl.zeros((K,), dtype=tl.float32)

    # Iterate over the filter weights
    for oc in range(K):
        oc_group = oc % groups
        ic_group = c % groups
        if oc_group != ic_group:
            continue

        for kh_group in range(KH):
            for kw_group in range(KW):
                weight_idx = oc * (C // groups) * KH * KW + ic_group * KH * KW + kh_group * KW + kw_group
                weight_val = tl.load(weight_ptr + weight_idx)

                # Compute the linear index within the input tensor for the current filter element
                in_filter_idx = n * C * padded_h * padded_w + c * padded_h * padded_w + (h_out - kh_group * dilation_h) * padded_w + (w_out - kw_group * dilation_w)

                # Load the input value for the current filter element
                input_filter_val = tl.load(input_ptr + in_filter_idx)

                # Accumulate the product of the input value and the weight value
                accum[oc] += input_val * weight_val * input_filter_val

    # Add the bias
    if bias_ptr is not None:
        bias_idx = oc
        accum += tl.load(bias_ptr + bias_idx)

    # Store the result
    out_idx = n * K * H * W + oc * H * W + h_out * W + w_out
    tl.store(output_ptr + out_idx, accum)

@triton.jit
def conv2d_add_backward(
    grad_output_ptr, input_ptr, weight_ptr, grad_input_ptr, grad_weight_ptr, grad_bias_ptr, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups, N, C, H, W, K, KH, KW
):
    """
    Backward pass for conv2d_add function.
    
    Parameters:
    - grad_output_ptr: Pointer to the gradient of the output tensor in global memory.
    - input_ptr: Pointer to the input tensor in global memory.
    - weight_ptr: Pointer to the weight tensor in global memory.
    - grad_input_ptr: Pointer to the gradient of the input tensor where the result will be stored.
    - grad_weight_ptr: Pointer to the gradient of the weight tensor where the result will be stored.
    - grad_bias_ptr: Pointer to the gradient of the bias tensor where the result will be stored.
    - stride_h, stride_w: Stride values for height and width.
    - padding_h, padding_w: Padding values for height and width.
    - dilation_h, dilation_w: Dilation values for height and width.
    - groups: Number of groups for grouped convolution.
    - N: Batch size.
    - C: Number of input channels.
    - H: Height of the input tensor.
    - W: Width of the input tensor.
    - K: Number of output channels.
    - KH: Height of the convolution kernel.
    - KW: Width of the convolution kernel.
    """
    n = tl.program_id(0)
    c = tl.program_id(1)
    kh = tl.program_id(2)
    kw = tl.program_id(3)

    # Compute the effective spatial dimensions after padding
    padded_h = H + 2 * padding_h
    padded_w = W + 2 * padding_w

    # Compute the spatial indices within the padded tensor
    h_out = n * stride_h + kh * dilation_h
    w_out = c * stride_w + kw * dilation_w

    # Clamp the spatial indices to ensure they are within bounds
    h_in = max(0, min(h_out, padded_h))
    w_in = max(0, min(w_out, padded_w))

    # Adjust the indices to account for padding
    h_in -= padding_h
    w_in -= padding_w

    # Compute the linear index within the input tensor
    in_idx = n * C * padded_h * padded_w + c * padded_h * padded_w + h_in * padded_w + w_in

    # Load the input value
    input_val = tl.load(input_ptr + in_idx)

    # Initialize the accumulation variable
    accum = tl.zeros((K,), dtype=tl.float32)

    # Iterate over the filter weights
    for oc in range(K):
        oc_group = oc % groups
        ic_group = c % groups
        if oc_group != ic_group:
            continue

        for kh_group in range(KH):
            for kw_group in range(KW):
                weight_idx = oc * (C // groups) * KH * KW + ic_group * KH * KW + kh_group * KW + kw_group
                weight_val = tl.load(weight_ptr + weight_idx)

                # Compute the linear index within the input tensor for the current filter element
                in_filter_idx = n * C * padded_h * padded_w + c * padded_h * padded_w + (h_out - kh_group * dilation_h) * padded_w + (w_out - kw_group * dilation_w)

                # Load the input value for the current filter element
                input_filter_val = tl.load(input_ptr + in_filter_idx)

                # Accumulate the product of the input value and the weight value
                accum[oc] += input_val * weight_val * input_filter_val

    # Add the bias
    if bias_ptr is not None:
        bias_idx = oc
        accum += tl.load(bias_ptr + bias_idx)

    # Store the result
    out_idx = n * K * H * W + oc * H * W + h_out * W + w_out
    tl.store(grad_output_ptr + out_idx, accum)

@triton.jit
def conv2d_add_forward(
    input_ptr, weight_ptr, bias_ptr, other_ptr, alpha, output_ptr, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups, N, C, H, W, K, KH, KW
):
    """
    Forward pass for conv2d_add function.
    
    Parameters:
    - input_ptr: Pointer to the input tensor in global memory.
    - weight_ptr: Pointer to the weight tensor in global memory.
    - bias_ptr: Pointer to the bias tensor in global memory.
    - other_ptr: Pointer to the other tensor or scalar in global memory.
    - alpha: Multiplier for other.
    - output_ptr: Pointer to the output tensor where the result will be stored.
    - stride_h, stride_w: Stride values for height and width.
    - padding_h, padding_w: Padding values for height and width.
    - dilation_h, dilation_w: Dilation values for height and width.
    - groups: Number of groups for grouped convolution.
    - N: Batch size.
    - C: Number of input channels.
    - H: Height of the input tensor.
    - W: Width of the input tensor.
    - K: Number of output channels.
    - KH
