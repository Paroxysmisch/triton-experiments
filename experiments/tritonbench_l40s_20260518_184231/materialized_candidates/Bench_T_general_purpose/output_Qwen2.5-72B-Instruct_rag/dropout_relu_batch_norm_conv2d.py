import triton
import triton.language as tl
import torch

# Triton kernel for 2D convolution
@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr, bias_ptr, stride, padding, dilation, groups,
    N, C_in, H, W, C_out, kH, kW, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // (H * W)
    h = (pid % (H * W)) // W
    w = (pid % (H * W)) % W

    # Load input and weight
    input = tl.load(input_ptr + n * C_in * H * W + tl.arange(0, C_in) * H * W + h * W + w)
    weight = tl.load(weight_ptr + tl.arange(0, C_out) * (C_in // groups) * kH * kW)

    # Perform convolution
    output = tl.zeros((C_out,), dtype=tl.float32)
    for g in range(groups):
        for kh in range(kH):
            for kw in range(kW):
                input_patch = tl.load(input_ptr + n * C_in * H * W + (g * (C_in // groups) + tl.arange(0, C_in // groups)) * H * W + (h + kh * dilation_h - padding_h) * W + (w + kw * dilation_w - padding_w)
                output += tl.dot(input_patch, weight[g * (C_in // groups) * kH * kW + kh * kW + kw])

    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr)
        output += bias

    # Store output
    tl.store(output_ptr + n * C_out * H * W + h * W + w, output)

# Triton kernel for batch normalization
@triton.jit
def batch_norm_kernel(
    input_ptr, output_ptr, gamma_ptr, beta_ptr, eps,
    N, C, H, W,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // (H * W)
    h = (pid % (H * W)) // W
    w = (pid % (H * W)) % W

    # Load input
    input = tl.load(input_ptr + n * C * H * W + h * W + w)

    # Load gamma and beta
    gamma = tl.load(gamma_ptr)
    beta = tl.load(beta_ptr)

    # Compute mean and variance
    mean = tl.mean(input, axis=1)
    var = tl.var(input, axis=1)

    # Normalize
    normalized = (input - mean) / tl.sqrt(var + eps)

    # Scale and shift
    output = normalized * gamma + beta

    # Store output
    tl.store(output_ptr + n * C * H * W + h * W + w, output)

# Triton kernel for ReLU activation
@triton.jit
def relu_kernel(
    input_ptr, output_ptr,
    N, C, H, W,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // (H * W)
    h = (pid % (H * W)) // W
    w = (pid % (H * W)) % W

    # Load input
    input = tl.load(input_ptr + n * C * H * W + h * W + w)

    # Apply ReLU
    output = tl.where(input > 0, input, 0)

    # Store output
    tl.store(output_ptr + n * C * H * W + h * W + w, output)

# Triton kernel for dropout
@triton.jit
def dropout_kernel(
    input_ptr, output_ptr, p, seed, size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    input = tl.load(input_ptr + offset, mask=mask)
    random = tl.rand(seed, offset)
    output = tl.where(random < p, 0, input / (1 - p))

    tl.store(output_ptr + offset, output, mask=mask)

# Combined Triton kernel for conv2d, batch norm, ReLU, and dropout
@triton.jit
def combined_kernel(
    input_ptr, weight_ptr, bias_ptr, gamma_ptr, beta_ptr, output_ptr, p, seed,
    stride, padding, dilation, groups, eps,
    N, C_in, H, W, C_out, kH, kW, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    size,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    # Convolution
    conv_output = tl.zeros((N, C_out, H, W), dtype=tl.float32)
    conv2d_kernel(
        input_ptr, weight_ptr, conv_output, bias_ptr, stride, padding, dilation, groups,
        N, C_in, H, W, C_out, kH, kW, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
        BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W
    )

    # Batch normalization
    bn_output = tl.zeros((N, C_out, H, W), dtype=tl.float32)
    batch_norm_kernel(
        conv_output, bn_output, gamma_ptr, beta_ptr, eps,
        N, C_out, H, W,
        BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W
    )

    # ReLU
    relu_output = tl.zeros((N, C_out, H, W), dtype=tl.float32)
    relu_kernel(
        bn_output, relu_output,
        N, C_out, H, W,
        BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W
    )

    # Dropout
    dropout_kernel(
        relu_output, output_ptr, p, seed, size,
        BLOCK_SIZE
    )
