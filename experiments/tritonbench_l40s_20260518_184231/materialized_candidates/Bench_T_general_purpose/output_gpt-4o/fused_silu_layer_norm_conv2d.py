import triton
import triton.language as tl

@triton.jit
def fused_silu_layer_norm_conv2d_kernel(
    x_ptr, conv_weight_ptr, conv_bias_ptr, weight_ptr, 
    output_ptr, stride, padding, dilation, groups, ln_eps, 
    H, W, C, KH, KW, stride_out, stride_in, stride_w, stride_b, stride_weight,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # Define the block indices
    bh = tl.program_id(0)
    bw = tl.program_id(1)

    # Define the starting point of the block
    h_start = bh * BLOCK_H
    w_start = bw * BLOCK_W

    # Load input and weights, apply convolution
    for h in range(BLOCK_H):
        for w in range(BLOCK_W):
            # Calculate the input position
            h_in = h_start + h * stride - padding
            w_in = w_start + w * stride - padding

            # Initialize convolution sum
            conv_sum = 0.0
            for kh in range(KH):
                for kw in range(KW):
                    for c in range(C):
                        # Check bounds
                        if 0 <= h_in + kh * dilation < H and 0 <= w_in + kw * dilation < W:
                            x_val = tl.load(x_ptr + (h_in + kh * dilation) * stride_in[1] + (w_in + kw * dilation) * stride_in[2] + c)
                            weight_val = tl.load(conv_weight_ptr + kh * stride_w[0] + kw * stride_w[1] + c)
                            conv_sum += x_val * weight_val

            # Add bias if provided
            if conv_bias_ptr:
                bias_val = tl.load(conv_bias_ptr + bw)
                conv_sum += bias_val

            # Apply layer normalization
            mean = conv_sum / (KH * KW * C)
            var = tl.sqrt(tl.sum((conv_sum - mean) ** 2) / (KH * KW * C) + ln_eps)
            normed = (conv_sum - mean) / var

            # Apply SiLU activation
            silu = normed * tl.sigmoid(normed)

            # Store result
            tl.store(output_ptr + (h_start + h) * stride_out[1] + (w_start + w) * stride_out[2], silu)

# Triton wrapper function
def fused_silu_layer_norm_conv2d(x, weight, conv_weight, conv_bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, ln_eps=1e-5):
    # Get dimensions
    B, C, H, W = x.shape
    OC, IC, KH, KW = conv_weight.shape

    # Prepare output tensor
    output = torch.empty((B, OC, H, W), device=x.device, dtype=x.dtype)

    # Calculate strides
    stride_in = x.stride()
    stride_out = output.stride()
    stride_w = conv_weight.stride()
    stride_b = conv_bias.stride() if conv_bias is not None else (0,)

    # Launch Triton kernel
    grid = (triton.cdiv(H, BLOCK_H), triton.cdiv(W, BLOCK_W))
    fused_silu_layer_norm_conv2d_kernel[grid](
        x, conv_weight, conv_bias, weight, output,
        conv_stride, conv_padding, conv_dilation, conv_groups, ln_eps,
        H, W, C, KH, KW, stride_out, stride_in, stride_w, stride_b, weight.stride(),
        BLOCK_H=8, BLOCK_W=8
    )

    return output
