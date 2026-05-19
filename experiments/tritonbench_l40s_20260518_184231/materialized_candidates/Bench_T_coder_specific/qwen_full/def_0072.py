import torch
import triton
import triton.language as tl

@triton.jit
def relu_batch_norm_conv2d_triton(x,
                                  weight,
                                  bias,
                                  stride,
                                  padding,
                                  dilation,
                                  groups,
                                  running_mean,
                                  running_var,
                                  bn_weight,
                                  bn_bias,
                                  is_train,
                                  momentum,
                                  eps,
                                  save_stats,
                                  x_mean,
                                  x_var,
                                  out,
                                  stride_x_c,
                                  stride_x_h,
                                  stride_x_w,
                                  stride_w_c,
                                  stride_w_kh,
                                  stride_w_kw,
                                  stride_b,
                                  stride_out_c,
                                  stride_out_h,
                                  stride_out_w,
                                  C,
                                  H,
                                  W,
                                  K,
                                  KH,
                                  KW,
                                  PAD_H,
                                  PAD_W,
                                  c1,
                                  c2,
                                  c3,
                                  c4,
                                  c5,
                                  c6,
                                  c7,
                                  c8,
                                  c9,
                                  c10,
                                  BLOCK_SIZE_C: tl.constexpr,
                                  BLOCK_SIZE_H: tl.constexpr,
                                  BLOCK_SIZE_W: tl.constexpr):
    # Positional indices
    c, h, w, b = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3)
    # The range of indices for each dimension is computed so we can index into the input matrix
    # using a single offset
    h_start = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_start = w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    c_tile_start = c * BLOCK_SIZE_C
    c_start = c_tile_start + tl.arange(0, BLOCK_SIZE_C)
    c_end = tl.minimum(c_start + BLOCK_SIZE_C, C) - 1
    h_in = h_start[:, None] + tl.arange(0, KH)[None, :]
    w_in = w_start[:, None] + tl.arange(0, KW)[None, :]
    c_in = c_start[None, :]
    c_k = c_tile_start + tl.arange(0, c10)
    # Load input and weights with proper padding
    x = tl.load(x + c_in * stride_x_c + h_in * stride_x_h + w_in * stride_x_w,
                mask=(c_start < C) & (h_in < PAD_H) & (w_in < PAD_W), other=0.0)
    w = tl.load(weight + c_k[None, :] * stride_w_c + (KW - 1 - tl.arange(0, c9))[
        None, :] * stride_w_kw + (KH - 1 - tl.arange(0, c8))[:, None] * stride_w_kh,
                mask=(c_k < C) & (tl.arange(0, c9) < KW) & (tl.arange(0, c8) < KH), other=0.0)
    # Apply convolution
    c_out = c * c7 + tl.arange(0, c6)
    out = tl.sum(x * w, 1)[:, None] + c5
    # Apply batch normalization
    x_mean = tl.sum(x, 1)[:, None] / (C * KH * KW)
    if is_train:
        x_var = tl.sum((x - x_mean) * (x - x_mean), 1)[:, None] / (C * KH * KW)
    else:
        x_var = tl.load(x_var + c_out * stride_out_c + h * stride_out_h +
                        w * stride_out_w, mask=(c_out < c4) & (h < c3) & (w < c2), other=0.0)
    x_hat = (x - x_mean) / tl.sqrt(x_var + eps)
    if save_stats:
        tl.store(running_mean + c_out * stride_out_c + h * stride_out_h +
                 w * stride_out_w, x_mean, mask=(c_out < c4) & (h < c3) & (w < c2))
        tl.store(running_var + c_out * stride_out_c + h * stride_out_h +
                 w * stride_out_w, x_var, mask=(c_out < c4) & (h < c3) & (w < c2))
    # Apply affine transformation
    if (bn_weight is not None) | (bn_bias is not None):
        gamma = tl.load(bn_weight + c_out,
                        mask=(c_out < c4), other=1.0).to(tl.float32)
        beta = tl.load(bn_bias + c_out, mask=(c_out < c4), other=0.0).to(tl.float32)
        out = x_hat * gamma + beta
    # Store output
    tl.store(out + c_out * stride_out_c + h * stride_out_h + w * stride_out_w,
             out, mask=(c_out < c4) & (h < c3) & (w < c2))


def relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, running_mean=None, running_var=None, bn_weight=None, bn_bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False) -> Tensor:
    if inplace:
        raise Exception("Inplace is not supported")

    has_bias = 1 if (bias is not None) else 0
    has_bn_weight = 1 if (bn_weight is not None) else 0
    has_bn_bias = 1 if (bn_bias is not None) else 0

    f = torch.empty_like(input, dtype=torch.float32)
    mean = torch.empty(input.shape[0], input.shape[2], input.shape[3],
                       dtype=torch.float32, device=input.device) if running_mean is None else running_mean
    var = torch.empty(input.shape[0], input.shape[2], input.shape[3],
                      dtype=torch.float32, device=input.device) if running_var is None else running_var

    # reshape input data into 2D tensor
    x_2d = as2d(input, groups)
    c, m, n = x_2d.shape
    # shape for final output
    out_shape = list(input.shape)
    # run kernel
    _relu_batch_norm_conv2d_triton[(m, n, c, 1)](
        x_2d,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        mean,
        var,
        bn_weight,
        bn_bias,
        training,
        momentum,
        eps,
        save_stats=has_bn_weight | has_bn_bias,
        x_mean=f,
        x_var=f,
        out=as2d(f, 1),
        stride_x_c=x_2d.stride(0),
        stride_x_h=x_2d.stride(1),
        stride_x_w=x_2d.stride(2),
        stride_w_c=weight.stride(0),
        stride_w_kh=weight.stride(1),
        stride_w_kw=weight.stride(2),
        stride_b=bias.stride(0) if has_bias else 0,
        stride_out_c=x_2d.stride(0),
        stride_out_h=x_2d.stride(1),
        stride_out_w=x_2d.stride(2),
        C=c, H=m, W=n, K=weight.shape[0], KH=weight.shape[1], KW=weight.shape[2],
        PAD_H=((stride - 1) * m + dilation * (KH - 1) - stride + 1),
        PAD_W=((stride - 1) * n + dilation * (KW - 1) - stride + 1),
        c1=triton.next_power_of_2(KW),
        c2=triton.next_power_of_2(KH),
        c3=triton.next_power_of_2(n),
        c4=triton.next_power_of_2(m),
        c5=torch.tensor(0.0, device=input.device),
        c6=triton.next_power_of_2(c),
        c7=triton.cdiv(c, triton.next_power_of_2(8)),
        c8=triton.next_power_of_2(KW),
        c9=triton.next_power_of_2(KH),
        c10=triton.next_power_of_2(groups),
        GROUP_SIZE=1,
        BLOCK_SIZE_C=triton.next_power_of_2(c),
        BLOCK_SIZE_H=triton.next_power_of_2(m),
        BLOCK_SIZE_W=triton.next_power_of_2(n),
        num_warps=4,
        num_stages=1,
    )
    return f.reshape(out_shape)


def _jit_wrapper_relu_batch_norm_conv2d(x, w, b, stride, padding, dilation, groups, running_mean, running_var, bn_weight, bn_bias, training, momentum, eps, inplace):
    return relu_batch_norm_conv2d(x, w, b, stride, padding, dilation, groups, running_mean, running_var, bn_weight, bn_bias, training, momentum, eps, inplace)
