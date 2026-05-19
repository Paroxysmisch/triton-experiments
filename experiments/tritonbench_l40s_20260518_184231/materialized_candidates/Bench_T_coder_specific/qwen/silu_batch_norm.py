import triton
import triton.language as tl

@triton.jit
def silu_batch_norm_kernel(
    X,
    Y,
    running_mean,
    running_var,
    weight,
    bias,
    N,
    C,
    H,
    W,
    stride_x,
    stride_y,
    stride_z,
    stride_w,
    stride_weight,
    stride_bias,
    stride_running_mean,
    stride_running_var,
    training,
    momentum,
    eps,
):
    # Calculate indices
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    # Compute the index in the flattened input tensor
    idx = n * stride_n + c * stride_c + h * stride_h + w * stride_w

    # Load the input data
    x = X[idx]

    # Batch normalization
    if training:
        mean = tl.reduce.mean(x, axis=[n], keepdim=True)
        var = tl.reduce.var(x, axis=[n], keepdim=True)
        running_mean[n, c] = running_mean[n, c] * (1 - momentum) + mean * momentum
        running_var[n, c] = running_var[n, c] * (1 - momentum) + var * momentum
    else:
        mean = running_mean[n, c]
        var = running_var[n, c]

    normalized_x = (x - mean) / tl.sqrt(var + eps)

    # Apply weight and bias if provided
    if weight is not None and bias is not None:
        normalized_x = normalized_x * weight[c] + bias[c]

    # Apply SiLU activation
    sielu_x = normalized_x * (1 / (1 + tl.exp(-normalized_x)))

    # Store the result
    Y[idx] = sielu_x

# Wrapper function
def silu_batch_norm(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-5
):
    # Get shape information
    N, C, H, W = input.shape
    stride_n, stride_c, stride_h, stride_w = input.stride

    # Create output tensor
    Y = input.zeros_like()

    # Launch the Triton kernel
    grid = (
        (N + 31) // 32,
        C,
        (H + 7) // 8,
        (W + 7) // 8,
    )
    block = (32, 1, 1)

    silu_batch_norm_kernel[grid, block](
        input.data_ptr(),
        Y.data_ptr(),
        running_mean.data_ptr(),
        running_var.data_ptr(),
        weight.data_ptr() if weight is not None else None,
        bias.data_ptr() if bias is not None else None,
        N,
        C,
        H,
        W,
        stride_n,
        stride_c,
        stride_h,
        stride_w,
        weight.stride[0] if weight is not None else 0,
        bias.stride[0] if bias is not None else 0,
        running_mean.stride[0],
        running_var.stride[0],
        training,
        momentum,
        eps,
    )

    return Y
