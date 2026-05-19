import torch
import triton
import triton.language as tl
from ssd.bi.softplus import softplus

@triton.jit
def fused_silu_layer_norm_conv2d_kernel(
    x_ptr,
    weight_ptr,
    conv_weight_ptr,
    conv_bias_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    C_out,
    conv_weight_h,
    conv_weight_w,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    ln_eps,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    stride_c_out,
    stride_conv_weight_h,
    stride_conv_weight_w,
    stride_conv_weight_c,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_c = tl.cdiv(C_out, BLOCK_SIZE_C)
    num_pid_h = tl.cdiv(H, BLOCK_SIZE_H)
    num_pid_w = tl.cdiv(W, BLOCK_SIZE_W)
    num_pid_in_group = GROUP_SIZE_M * num_pid_c
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_M
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_M)
    pid_n = first_pid_n + (pid % group_size_n)
    pid_c = (pid % num_pid_in_group) // group_size_n
    pid_h = (pid // num_pid_c) % num_pid_h
    pid_w = (pid % num_pid_c) // num_pid_h

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_conv_weight_h = tl.arange(0, BLOCK_SIZE_CONV_WEIGHT_H)
    offs_conv_weight_w = tl.arange(0, BLOCK_SIZE_CONV_WEIGHT_W)

    x_base_ptr = x_ptr + (
        offs_n[:, None, None] * stride_n + offs_c[None, :, None] * stride_c
    )
    x = tl.load(
        x_base_ptr,
        mask=(offs_n[:, None, None] < N) & (offs_c[None, :, None] < C),
        other=0.0,
    )
    conv_weight_base_ptr = conv_weight_ptr + (
        offs_conv_weight_h[:, None, None] * stride_conv_weight_h
        + offs_conv_weight_w[None, :, None] * stride_conv_weight_w
        + offs_c[None, None, :] * stride_conv_weight_c
    )
    conv_weight = tl.load(
        conv_weight_base_ptr,
        mask=(offs_conv_weight_h[:, None, None] < conv_weight_h)
        & (offs_conv_weight_w[None, :, None] < conv_weight_w)
        & (offs_c[None, None, :] < C_out),
        other=0.0,
    ).to(tl.float32)

    conv_bias = tl.load(conv_bias_ptr).to(tl.float32)

    conv_out = tl.dot(x, conv_weight) + conv_bias
    conv_out = tl.reshape(conv_out, (BLOCK_SIZE_N, -1, BLOCK_SIZE_W))
    conv_out = tl.transpose(conv_out, (0, 2, 1))
    conv_out = tl.reshape(conv_out, (N, -1))

    mean = tl.sum(conv_out, axis=1) / (N * C_out)
    conv_out = tl.where((offs_n[:, None, None] < N) & (offs_c[None, :, None] < C_out),
                    conv_out,
                    0.0)
    conv_out = tl.where((offs_n[:, None, None] < N) & (offs_c[None, :, None] < C_out),
                    conv_out - mean[:, None],
                    0.0)
    conv_out_var = tl.sum(conv_out * conv_out, axis=1) / (N * C_out)
    rstd = 1 / tl.sqrt(conv_out_var + ln_eps)

    weight = tl.load(weight_ptr + offs_c, mask=offs_c < C_out, other=0.0)

    out = conv_out * (weight * rstd)[:, None]
    out = softplus(out)
    out_base_ptr = out_ptr + (
        offs_n[:, None, None] * stride_n
        + (offs_c + C_out)[None, :, None] * stride_c
    )
    tl.store(
        out_base_ptr,
        out,
        mask=(offs_n[:, None, None] < N) & (offs_c[None, :, None] < C_out),
    )


def fused_silu_layer_norm_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor = None,
    conv_stride: int = 1,
    conv_padding: int = 0,
    conv_dilation: int = 1,
    conv_groups: int = 1,
    ln_eps: float = 1e-5,
) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert weight.dtype == torch.float16
    assert conv_weight.dtype == torch.float16
    assert x.shape[1] == weight.shape[0]
    assert (
        conv_weight.shape[1] == x.shape[1]
    ), f"{conv_weight.shape=} must have the same number of input channels as x has features"
    assert conv_bias is None or conv_bias.shape == (conv_weight.shape[0],)
    assert 0 <= conv_padding <= conv_dilation
    assert conv_weight.shape[2] == conv_weight.shape[3]
    conv_weight_h = conv_weight.shape[2]
    conv_weight_w = conv_weight.shape[3]
    assert conv_weight_h % 2 == 1 and conv_weight_w % 2 == 1

    batch_size, in_channels, in_height, in_width = x.shape
    out_channels = weight.shape[0]
    out_height = triton.cdiv(in_height, conv_stride)
    out_width = triton.cdiv(in_width, conv_stride)

    conv_weight = conv_weight.contiguous()
    if conv_bias is not None:
        conv_bias = conv_bias.contiguous()
    weight = weight.contiguous()

    out = torch.empty(
        (batch_size, out_channels, out_height, out_width),
        device=x.device,
        dtype=x.dtype,
    )

    grid = lambda META: (
        triton.cdiv(in_height, META["BLOCK_SIZE_H"])
        * triton.cdiv(in_width, META["BLOCK_SIZE_W"])
        * triton.cdiv(in_channels, META["BLOCK_SIZE_C"]),
    )
    BLOCK_SIZE_N = 32
    GROUP_SIZE_M = 8
    (
        BLOCK_SIZE_CONV_WEIGHT_H,
        BLOCK_SIZE_CONV_WEIGHT_W,
    ) = conv_weight.shape[-2:]
    num_warps = 4
    pad_h = (
        (conv_weight_h - conv_stride) // 2 + conv_padding
    )  # (kH - stride) // 2 + padding
    pad_w = (
        (conv_weight_w - conv_stride) // 2 + conv_padding
    )  # (kH - stride) // 2 + padding
    x = tl.zeros(
        ([1, 1, pad_h * 2 + in_height, pad_w * 2 + in_width]) + x.shape[1:],
        dtype=x.dtype,
    )
    x[:, :, pad_h:pad_h + in_height, pad_w:pad_w + in_width] = x
    x = x.to(x.dtype)

    with torch.cuda.device(x.device.index):
        fused_silu_layer_norm_conv2d_kernel[grid](
            x,
            weight,
            conv_weight,
            conv_bias,
            out,
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            conv_weight_h,
            conv_weight_w,
            conv_stride,
            pad_h,
            conv_dilation,
            conv_groups,
            ln_eps,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            conv_weight.stride(0),
            conv_weight.stride(1),
            conv_weight.stride(2),
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            GROUP_SIZE_M=GROUP_SIZE_M,
            BLOCK_SIZE_CONV_WEIGHT_H=BLOCK_SIZE_CONV_WEIGHT_H,
            BLOCK_SIZE_CONV_WEIGHT_W=BLOCK_SIZE_CONV_WEIGHT_W,
            num_warps=num_warps,
        )
    return out
