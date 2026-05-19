import triton
import triton.language as tl

@triton.jit
def fused_silu_layer_norm_conv2d_kernel(
    x_ptr, 
    weight_ptr, 
    conv_weight_ptr, 
    conv_bias_ptr,
    out_ptr,
    stride, 
    padding, 
    dilation, 
    groups,
    ln_eps,
    N, C, H, W, CI, CO, KH, KW,
    BLOCK_SIZE=64):
    
    pid = tl.program_id(axis=0)
    X = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    Y = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load x
    x = tl.load(x_ptr + X[:, None] * (H + 2 * padding) * W + Y[None, :])

    # Convolution
    conv_out = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=x.dtype)
    for kh in range(KH):
        for kw in range(KW):
            pad_h = kh - padding
            pad_w = kw - padding
            if pad_h >= 0 and pad_h < H and pad_w >= 0 and pad_w < W:
                conv_out += tl.dot(x[pad_h:pad_h+stride, pad_w:pad_w+stride], conv_weight[kh*KW+kw])

    if conv_bias_ptr is not None:
        conv_out += conv_bias

    # Layer Normalization
    mean = tl.mean(conv_out, axis=1)
    var = tl.variance(conv_out, axis=1)
    normed = (conv_out - mean[:, None]) / tl.sqrt(var[:, None] + ln_eps)
    normed *= weight

    # SiLU Activation
    silu_out = normed * (normed > 0)

    # Store result
    tl.store(out_ptr + X[:, None] * (H + 2 * padding) * W + Y[None, :], silu_out)

@triton.autotune
def fused_silu_layer_norm_conv2d(
    x: torch.Tensor, 
    weight: torch.Tensor, 
    conv_weight: torch.Tensor, 
    conv_bias: torch.Tensor = None, 
    conv_stride: int = 1, 
    conv_padding: int = 0, 
    conv_dilation: int = 1, 
    conv_groups: int = 1, 
    ln_eps: float = 1e-5) -> torch.Tensor:
    
    N, C, H, W = x.shape
    CI, CO, KH, KW = conv_weight.shape
    
    assert C == CI * conv_groups, "Input channels must match the number of groups times the group's input channels"
    assert CO % conv_groups == 0, "Output channels must be divisible by the number of groups"
    
    y = torch.empty_like(x)
    
    grid = lambda META: (
        (N + META.BLOCK_SIZE - 1) // META.BLOCK_SIZE,
        (H + 2 * conv_padding + META.BLOCK_SIZE - 1) // META.BLOCK_SIZE,
        (W + 2 * conv_padding + META.BLOCK_SIZE - 1) // META.BLOCK_SIZE,
    )
    
    fused_silu_layer_norm_conv2d_kernel[
        grid,
        {
            "BLOCK_SIZE": 64,
        }
    ](
        x.data_ptr(),
        weight.data_ptr(),
        conv_weight.data_ptr(),
        conv_bias.data_ptr() if conv_bias is not None else 0,
        y.data_ptr(),
        conv_stride,
        conv_padding,
        conv_dilation,
        conv_groups,
        ln_eps,
        N, C, H, W, CI, CO, KH, KW
    )

    return y
