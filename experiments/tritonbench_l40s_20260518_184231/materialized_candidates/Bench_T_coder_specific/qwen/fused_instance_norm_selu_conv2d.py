import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    stride,
    padding,
    dilation,
    groups,
    N,
    C_in,
    H_in,
    W_in,
    C_out,
    K,
    pad_H,
    pad_W,
    stride_H,
    stride_W,
    dilation_H,
    dilation_W,
):
    row = tl.program_id(0)
    col = tl.program_id(1)

    h_start = row * stride_H - pad_H
    w_start = col * stride_W - pad_W

    out_c = row // stride_H
    in_c = row % stride_H
    out_h = col // stride_W
    in_w = col % stride_W

    acc = tl.zeros((K,), dtype=tl.float32)
    for kh in range(K):
        for kw in range(K):
            ih = h_start + kh * dilation_H
            iw = w_start + kw * dilation_W
            if 0 <= ih < H_in and 0 <= iw < W_in:
                acc[kh] += input_ptr[(ih, iw, in_c)] * weight_ptr[(kh, kw, in_c)]

    output_ptr[(row, col)] = acc.sum()
