import torch
import triton
import triton.language as tl

@triton.jit
def dropout_relu_batch_norm_conv2d(
    input, weight, bias, stride, padding, dilation, groups, p, training, inplace
):
    N, C, H, W = tl.shape(input)
    C1, C2, kH, kW = tl.shape(weight)
    assert C == C2
    assert C % groups == 0
    assert C1 % groups == 0
    N = N
    C = C
    G = groups
    if isinstance(padding, str):
        padding = [padding, padding]
    if isinstance(stride, int):
        stride = [stride, stride]
    if isinstance(dilation, int):
        dilation = [dilation, dilation]
    pad_h = padding[0]
    pad_w = padding[1]
    stride_h = stride[0]
    stride_w = stride[1]
    dilation_h = dilation[0]
    dilation_w = dilation[1]
    out_h = tl.cdiv(H, stride_h)
    out_w = tl.cdiv(W, stride_w)
    # out
    output = tl.zeros([N, C1, out_h, out_w], dtype=tl.float32)
    # patch index
    ph = tl.arange(0, kH)
    pw = tl.arange(0, kW)
    # stride
    input_stride_n = C * H * W
    input_stride_c = H * W
    input_stride_h = W
    input_stride_w = 1
    weight_stride_c = kH * kW
    weight_stride_k = kW
    weight_stride_d = 1

    # [N, C, H, W]
    input_offsets_n = tl.arange(0, 32)
    input_offsets_c = tl.arange(0, 32)
    input_offsets_h = tl.arange(0, 8) * stride_h + ph[None, :] * dilation_h
    input_offsets_w = tl.arange(0, 8) * stride_w + pw[None, :] * dilation_w
    input_base = (
        input
        + input_offsets_n[:, None, None] * input_stride_n
        + input_offsets_c[None, :, None] * input_stride_c
    )

    # [C, C1, kH, kW]
    weight_offsets_c = tl.arange(0, 8)[:, None] * weight_stride_c
    weight_offsets_k = tl.arange(0, 8)[None, :] * weight_stride_k
    weight_base = (
        weight
        + weight_offsets_c
        + weight_offsets_k
        + (ph[:, None] * dilation_h + pw[None, :] * dilation_w) * weight_stride_d
    )

    # [N, C1, out_h, out_w]
    output_offsets_n = tl.arange(0, 32)
    output_offsets_c = tl.arange(0, 32)
    output_offsets_h = tl.arange(0, 8) * stride_h
    output_offsets_w = tl.arange(0, 8) * stride_w
    output_base = (
        output
        + output_offsets_n[:, None, None] * N * C1 * out_h * out_w
        + output_offsets_c[None, :, None] * out_h * out_w
        + output_offsets_h[:, None, None] * out_w
        + output_offsets_w[None, :, None]
    )

    # compute
    for n in range(0, N):
        for h in range(0, out_h):
            for w in range(0, out_w):
                # [8, 8]
                input_index = (
                    input_base
                    + input_offsets_h[:, None]
                    + input_offsets_w[None, :]
                    + h * stride_h * input_stride_h
                    + w * stride_w * input_stride_w
                )
                input_mask = (
                    (input_index < (N * C * H * W))
                    & ((input_offsets_h[:, None] + h * stride_h) < H)
                    & ((input_offsets_w[None, :] + w * stride_w) < W)
                )
                input_ptrs = input + input_index
                input_val = tl.load(input_ptrs, mask=input_mask)

                # [8, 8, 32]
                weight_index = (
                    weight_base
                    + (n % G) * (C // G) * kH * kW
                    + ((n // G) * (C1 // G)) * weight_stride_c
                )
                weight_mask = (
                    ((n % G) * (C // G) + weight_offsets_c[:, None])
                    < C
                    and ((n // G) * (C1 // G) + weight_offsets_k[None, :]) < C1
                )
                weight_ptrs = weight + weight_index
                weight_val = tl.load(weight_ptrs, mask=weight_mask)

                # [8, 8, 32]
                acc = tl.dot(input_val, weight_val, allow_tf32=False)

                if bias is not None:
                    # [1, 32]
                    bias_index = (
                        bias
                        + (n % G) * (C // G)
                        + ((n // G) * (C1 // G)) * bias_stride
                    )
                    bias_mask = (
                        ((n % G) * (C // G) + bias_offsets) < C
                        and ((n // G) * (C1 // G) + bias_offsets) < C1
                    )
                    bias_ptrs = bias + bias_index
                    bias_val = tl.load(bias_ptrs, mask=bias_mask)
                    acc += bias_val

                # [32, 32]
                acc = tl.where(
                    output_offsets_h[:, None] + h * stride_h < H,
                    acc,
                    0,
                )
                acc = tl.where(
                    output_offsets_w[None, :] + w * stride_w < W,
                    acc,
                    0,
                )

                if not inplace:
                    # [32, 32]
                    output_ptrs = output_base + h * out_w * C1 + w * C1
                    tl.store(output_ptrs, acc)

    if inplace:
        return output.to(input.dtype)
    else:
        # dropout
        keep_mask = tl.rand(input) > p
        output = tl.where(keep_mask, output / (1 - p), 0)
        return output.to(input.dtype)
