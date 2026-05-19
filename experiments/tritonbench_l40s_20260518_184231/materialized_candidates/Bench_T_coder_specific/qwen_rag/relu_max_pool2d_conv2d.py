@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr, bias_ptr, padding, dilation, stride, out_shape, N, C, H, W, K, R, S, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(N * C * H * W, BLOCK_SIZE)
    if pid >= grid_size:
        return

    n = pid // (C * H * W)
    c = (pid // (H * W)) % C
    h = (pid // W) % H
    w = pid % W

    h_start = h * stride - padding
    w_start = w * stride - padding

    acc = 0.0
    for kh in range(R):
        for kw in range(S):
            ih = h_start + kh * dilation
            iw = w_start + kw * dilation
            if 0 <= ih < H and 0 <= iw < W:
                ic = c * dilation + kh * dilation + kw
                acc += tl.load(input_ptr + n * C * H * W + c * H * W + ih * W + iw) * tl.load(weight_ptr + c * R * S + kh * S + kw)

    if bias_ptr is not None:
        acc += tl.load(bias_ptr + c)

    tl.store(output_ptr + n * C * out_shape[2] * out_shape[3] + c * out_shape[2] * out_shape[3] + h * out_shape[3] + w, acc)
