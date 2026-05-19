import torch
import triton
import triton.language as tl

# Triton kernel for 2D convolution
@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, other_ptr, output_ptr,
    B, C, H, W, O, K, R, S, stride_h, stride_w, padding_h, padding_w,
    dilation_h, dilation_w, groups, alpha, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE_N
    block_start = pid * block_size

    # Compute the output dimensions
    out_h = (H + 2 * padding_h - dilation_h * (R - 1) - 1) // stride_h + 1
    out_w = (W + 2 * padding_w - dilation_w * (S - 1) - 1) // stride_w + 1

    # Iterate over the output elements
    for n in range(block_start, block_start + block_size):
        if n >= B * out_h * out_w:
            break

        b = n // (out_h * out_w)
        out_h_idx = (n % (out_h * out_w)) // out_w
        out_w_idx = n % out_w

        in_h_start = out_h_idx * stride_h - padding_h
        in_w_start = out_w_idx * stride_w - padding_w

        output_val = tl.zeros((O,), dtype=tl.float32)

        for g in range(groups):
            for k in range(K // groups):
                for r in range(R):
                    for s in range(S):
                        in_h = in_h_start + r * dilation_h
                        in_w = in_w_start + s * dilation_w

                        if 0 <= in_h < H and 0 <= in_w < W:
                            in_c_start = g * (C // groups)
                            in_c_end = (g + 1) * (C // groups)

                            weight_val = tl.load(weight_ptr + g * (K // groups) * R * S + k * R * S + r * S + s)
                            input_val = tl.load(input_ptr + b * C * H * W + in_c_start * H * W + in_h * W + in_w)

                            output_val += input_val * weight_val

        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + n % O)
            output_val += bias_val

        if other_ptr is not None:
            other_val = tl.load(other_ptr + n % O)
            output_val += alpha * other_val

        tl.store(output_ptr + n, output_val)

# Wrapper function for 2D convolution
def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    # Validate input shapes and parameters
    B, C, H, W = input.shape
    O, C_per_group, R, S = weight.shape
    assert C % groups == 0, "in_channels must be divisible by groups"
    assert O % groups == 0, "out_channels must be divisible by groups"

    # Handle stride, padding, and dilation
    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int) or padding == 'valid' or padding == 'same':
        if padding == 'valid':
            padding_h, padding_w = 0, 0
        elif padding == 'same':
            padding_h = (dilation_h * (R - 1) + 1 - H % stride_h) // 2
            padding_w = (dilation_w * (S - 1) + 1 - W % stride_w) // 2
        else:
            padding_h, padding_w = padding, padding
    else:
        padding_h, padding_w = padding

    if isinstance(dilation, int):
        dilation_h, dilation_w = dilation, dilation
    else:
        dilation_h, dilation_w = dilation

    # Compute output dimensions
    out_h = (H + 2 * padding_h - dilation_h * (R - 1) - 1) // stride_h + 1
    out_w = (W + 2 * padding_w - dilation_w * (S - 1) - 1) // stride_w + 1

    # Initialize output tensor
    if out is None:
        out = torch.empty((B, O, out_h, out_w), device=input.device, dtype=input.dtype)

    # Flatten the input and output tensors for Triton
    input_flat = input.view(B, C, H * W)
    weight_flat = weight.view(O, C_per_group, R * S)
    out_flat = out.view(B * O, out_h * out_w)

    # Flatten bias and other if they are provided
    if bias is not None:
        bias_flat = bias.view(O)
    else:
        bias_flat = None

    if other is not None:
        other_flat = other.view(O)
    else:
        other_flat = None

    # Launch the Triton kernel
    BLOCK_SIZE_N = 256
    num_blocks = (B * out_h * out_w + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (num_blocks,)

    conv2d_kernel[grid](
        input_flat, weight_flat, bias_flat, other_flat, out_flat,
        B, C, H, W, O, K, R, S, stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w, groups, alpha, BLOCK_SIZE_N
    )

    return out
