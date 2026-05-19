import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C_in, H, W, C_out, K_h, K_w,
    stride_h, stride_w, pad_h, pad_w,
    dilation_h, dilation_w, groups,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    batch_id = tl.program_id(0)
    out_h = (H + 2 * pad_h - dilation_h * (K_h - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dilation_w * (K_w - 1) - 1) // stride_w + 1
    oh = tl.program_id(1)
    ow = tl.arange(0, BLOCK_W)

    oh_idx = oh * BLOCK_H + tl.arange(0, BLOCK_H)
    ow_idx = ow * BLOCK_W + tl.arange(0, BLOCK_W)

    for oc in range(C_out):
        result = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
        for ic in range(C_in // groups):
            for kh in range(K_h):
                for kw in range(K_w):
                    ih = oh_idx * stride_h + kh * dilation_h - pad_h
                    iw = ow_idx * stride_w + kw * dilation_w - pad_w
                    mask = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                    input_val = tl.load(input_ptr + batch_id * C_in * H * W + ic * H * W + ih * W + iw, mask=mask, other=0.0)
                    weight_val = tl.load(weight_ptr + oc * (C_in // groups) * K_h * K_w + ic * K_h * K_w + kh * K_w + kw)
                    result += input_val * weight_val
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + oc)
            result += bias_val
        result = 1 / (1 + tl.exp(-result))
        tl.store(output_ptr + batch_id * C_out * out_h * out_w + oc * out_h * out_w + oh_idx * out_w + ow_idx, result)

def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    assert input.device.type == 'cuda', "Input tensor must be on a CUDA device for Triton ops."
    assert input.device == weight.device, "Input and weight must be on the same CUDA device."
    if bias is not None:
        assert input.device == bias.device, "Input and bias must be on the same CUDA device."

    B, C_in, H, W = input.shape
    C_out, _, K_h, K_w = weight.shape

    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation

    out_h = (H + 2 * pad_h - dilation_h * (K_h - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dilation_w * (K_w - 1) - 1) // stride_w + 1

    if out is None:
        out = torch.empty((B, C_out, out_h, out_w), device=input.device, dtype=input.dtype)

    BLOCK_H = 8
    BLOCK_W = 8
    grid = (B, (out_h + BLOCK_H - 1) // BLOCK_H, (out_w + BLOCK_W - 1) // BLOCK_W)

    conv2d_kernel[grid](
        input_ptr=input, weight_ptr=weight, bias_ptr=bias, output_ptr=out,
        B=B, C_in=C_in, H=H, W=W, C_out=C_out, K_h=K_h, K_w=K_w,
        stride_h=stride_h, stride_w=stride_w, pad_h=pad_h, pad_w=pad_w,
        dilation_h=dilation_h, dilation_w=dilation_w, groups=groups,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    return out
