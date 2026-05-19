import triton
from triton.language import *
from triton.runtime import *

@triton.jit
def conv2d_add_kernel(
    input,
    weight,
    bias,
    other,
    output,
    minb: tl.int32,
    ic: tl.int32,
    ih: tl.int32,
    iw: tl.int32,
    oc: tl.int32,
    kh: tl.int32,
    kw: tl.int32,
    stride_h: tl.int32,
    stride_w: tl.int32,
    pad_h: tl.int32,
    pad_w: tl.int32,
    dilation_h: tl.int32,
    dilation_w: tl.int32,
    groups: tl.int32,
    alpha: tl.float32):
    pid = tl.program_id(axis=0)
    bid = pid // (oc // groups)
    c = pid % (oc // groups)

    h_start = pid // (ic // groups) * stride_h - pad_h
    w_start = pid % (ic // groups) * stride_w - pad_w

    @tl.primfn
    def load_shared_input(ih, iw):
        in_idx = bid * ic * ih * iw + c * ih * iw + (h_start + ih * dilation_h) * iw + (w_start + iw * dilation_w)
        sh_idx = ih * 32 + iw
        if (h_start + ih * dilation_h >= 0) and (h_start + ih * dilation_h < ih) and (w_start + iw * dilation_w >= 0) and (w_start + iw * dilation_w < iw):
            return input[in_idx]
        else:
            return 0.0

    @tl.primfn
    def perform_convolution():
        sum = 0.0
        for ih in range(kh):
            for iw in range(kw):
                sh_idx = ih * 32 + iw
                sum += load_shared_input(ih, iw) * weight[c * kh * kw + ih * kw + iw]
        return sum

    @tl.primfn
    def add_bias(sum):
        if bias is not None:
            sum += bias[c]
        return sum

    @tl.primfn
    def add_other_scaled_by_alpha(sum):
        if other is not None:
            sum += alpha * other[bid * oc + c]
        return sum

    @tl.primfn
    def store_result(sum):
        oh = pid // (ic // groups) * stride_h - pad_h + kh // 2
        ow = pid % (ic // groups) * stride_w - pad_w + kw // 2
        if (oh >= 0) and (oh < ih) and (ow >= 0) and (ow < iw):
            output[bid * oc * ih * iw + c * ih * iw + oh * iw + ow] = sum

    sum = perform_convolution()
    sum = add_bias(sum)
    sum = add_other_scaled_by_alpha(sum)
    store_result(sum)

@triton.autotune
def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    if isinstance(stride, int):
        stride_h = stride_w = stride
    elif isinstance(stride, tuple):
        stride_h, stride_w = stride
    else:
        raise ValueError("stride must be an integer or a tuple")

    if isinstance(padding, str):
        if padding == 'valid':
            pad_h = pad_w = 0
        elif padding == 'same':
            pad_h = (kH - 1) // 2
            pad_w = (kW - 1) // 2
        else:
            raise ValueError("padding must be 'valid', 'same', a single number, or a tuple")
    elif isinstance(padding, int):
        pad_h = pad_w = padding
    elif isinstance(padding, tuple):
        pad_h, pad_w = padding
    else:
        raise ValueError("padding must be an integer or a tuple")

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    elif isinstance(dilation, tuple):
        dilation_h, dilation_w = dilation
    else:
        raise ValueError("dilation must be an integer or a tuple")

    if out is None:
        out = torch.zeros((minibatch, out_channels, iH, iW), dtype=input.dtype, device=input.device)

    # Launch the Triton kernel
    grid = (minibatch, out_channels // groups
