import triton
import triton.language as tl

@triton.jit
def conv2d_forward_kernel(
    Input, Weight, Output, InputPad, Bias, Stride, Pad, Dilation, Groups, N, C, H, W, K, KH, KW,
    BlockSize: tl.constexpr, GroupSize: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row = pid % (N * (H + 2 * Pad[0]) // Stride[0])
    col = pid // (N * (H + 2 * Pad[0]) // Stride[0])

    n = row // ((H + 2 * Pad[0]) // Stride[0])
    h_out = row % ((H + 2 * Pad[0]) // Stride[0])
    w_out = col % ((W + 2 * Pad[1]) // Stride[1])
    c_in_group = col // ((W + 2 * Pad[1]) // Stride[1])

    acc = tl.zeros((GroupSize,), dtype=tl.float32)
    h_start = h_out * Stride[0] - Pad[0]
    w_start = w_out * Stride[1] - Pad[1]

    for kh in range(KH):
        for kw in range(KW):
            h_in = h_start + kh * Dilation[0]
            w_in = w_start + kw * Dilation[1]
            if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                c_in = c_in_group * (KH * KW) + kh * KW + kw
                c_out_group = (c_in_group // (C // Groups)) * Groups
                acc[c_in_group] += Input[n, c_in_group, h_in, w_in] * Weight[c_out_group, c_in_group, kh, kw]

    if Bias is not None:
        acc += Bias[c_in_group]

    acc = tl.max(acc, tl.zeros((GroupSize,), dtype=tl.float32))
    acc = tl.min(acc, tl.ones((GroupSize,), dtype=tl.float32))

    Output[row, col] = acc
