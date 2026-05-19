import triton
import triton.language as tl
import torch

@triton.jit
def conv2d_forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, C_in, H_in, W_in,
    C_out,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    groups,
    in_strdB, in_strdC, in_strdH, in_strdW,
    w_strdO, w_strdI, w_strdH, w_strdW,
    out_strdB, out_strdC, out_strdH, out_strdW,
    BLOCK_B: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    USE_FP16: tl.constexpr,
    USE_TF32: tl.constexpr
):
    b_block_id = tl.program_id(0)
    co_block_id = tl.program_id(1)
    oh_block_id = tl.program_id(2)

    b_range = b_block_id * BLOCK_B + tl.arange(0, BLOCK_B)
    co_range = co_block_id * BLOCK_C_OUT + tl.arange(0, BLOCK_C_OUT)
    oh = oh_block_id
    W_out = (W_in + 2 * pad_w - KW) // stride_w + 1

    b_mask = b_range < B
    co_mask = co_range < C_out

    for ow in range(0, W_out):
        acc = tl.zeros((BLOCK_B, BLOCK_C_OUT), dtype=tl.float32)
        c_in_group = C_in // groups
        co_per_group = C_out // groups
        group_id = co_range // co_per_group
        c_in_start = group_id * c_in_group

        for ci_offset in range(c_in_group):
            ci = c_in_start + ci_offset
            for kh in range(KH):
                for kw in range(KW):
                    ih = oh * stride_h - pad_h + kh
                    iw = ow * stride_w - pad_w + kw
                    in_bounds = (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in)
                    if in_bounds:
                        inp_offset = b_range * in_strdB + ci * in_strdC + ih * in_strdH + iw * in_strdW
                        w_offset = co_range * w_strdO + ci * w_strdI + kh * w_strdH + kw * w_strdW
                        inp = tl.load(input_ptr + inp_offset, mask=b_mask, other=0.0)
                        wgt = tl.load(weight_ptr + w_offset, mask=co_mask, other=0.0)
                        acc += inp * wgt

        out_offset = b_range * out_strdB + co_range * out_strdC + oh * out_strdH + ow * out_strdW
        out_val = acc
        if USE_FP16:
            out_val = out_val.to(tl.float16)
        tl.store(output_ptr + out_offset, out_val, mask=(b_mask[:, None] & co_mask[None, :]))


def conv2d_forward(input, weight, kernel_size, stride, padding, groups=1, use_fp16=False, use_tf32=False):
    B, C_in, H_in, W_in = input.shape
    C_out, _, KH, KW = weight.shape

    out_height = (H_in + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    out_width = (W_in + 2 * padding[1] - kernel_size[1]) // stride[1] + 1

    dtype = torch.float16 if use_fp16 else torch.float32
    output = torch.zeros((B, C_out, out_height, out_width), dtype=dtype, device=input.device)

    in_strdB = input.stride(0)
    in_strdC = input.stride(1)
    in_strdH = input.stride(2)
    in_strdW = input.stride(3)
    w_strdO = weight.stride(0)
    w_strdI = weight.stride(1)
    w_strdH = weight.stride(2)
    w_strdW = weight.stride(3)
    out_strdB = output.stride(0)
    out_strdC = output.stride(1)
    out_strdH = output.stride(2)
    out_strdW = output.stride(3)

    BLOCK_B = 1
    BLOCK_C_OUT = 1
    BLOCK_C_IN = 1

    grid = (
        (B + BLOCK_B - 1) // BLOCK_B,
        (C_out + BLOCK_C_OUT - 1) // BLOCK_C_OUT,
        out_height
    )

    conv2d_forward_kernel[grid](
        input, weight, output,
        B, C_in, H_in, W_in,
        C_out,
        KH, KW,
        stride[0], stride[1],
        padding[0], padding[1],
        groups,
        in_strdB, in_strdC, in_strdH, in_strdW,
        w_strdO, w_strdI, w_strdH, w_strdW,
        out_strdB, out_strdC, out_strdH, out_strdW,
        BLOCK_B=BLOCK_B,
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_C_IN=BLOCK_C_IN,
        USE_FP16=use_fp16,
        USE_TF32=use_tf32
    )

    return output
