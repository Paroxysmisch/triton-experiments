import torch
import triton
import triton.language as tl

@triton.jit
def fused_relu_fractional_max_pool2d_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    input_n, input_c, input_h, input_w,
    kernel_h, kernel_w,
    output_h, output_w,
    stride_h_scale, stride_w_scale,
    return_indices: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(input_n * input_c, BLOCK_SIZE_H * BLOCK_SIZE_W)
    pid_nc = pid // (output_h * output_w)
    pid_hw = pid % (output_h * output_w)
    pid_out_h = pid_hw // output_w
    pid_out_w = pid_hw % output_w

    nc = pid_nc * BLOCK_SIZE_H * BLOCK_SIZE_W
    off_n = nc // input_c
    off_c = nc % input_c

    if off_n >= input_n:
        return

    start_h = pid_out_h * stride_h_scale
    start_w = pid_out_w * stride_w_scale

    start_h_int = tl.math.floor(start_h).to(tl.int32)
    start_w_int = tl.math.floor(start_w).to(tl.int32)

    max_val = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32) - float('inf')
    max_idx = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.int64)

    for kh in range(kernel_h):
        for kw in range(kernel_w):
            h_in = start_h_int + kh
            w_in = start_w_int + kw
            if h_in < input_h and w_in < input_w:
                for batch in range(BLOCK_SIZE_H):
                    for channel in range(BLOCK_SIZE_W):
                        n = off_n + batch
                        c = off_c + channel
                        if n < input_n and c < input_c:
                            offset = n * input_c * input_h * input_w + c * input_h * input_w + h_in * input_w + w_in
                            val = tl.load(input_ptr + offset)
                            val = tl.maximum(val, 0.0)
                            if val > max_val[batch, channel]:
                                max_val = tl.where(tl.arange(BLOCK_SIZE_H)[:, None] == batch & tl.arange(BLOCK_SIZE_W)[None, :] == channel, val, max_val)
                                idx = h_in * input_w + w_in
                                max_idx = tl.where(tl.arange(BLOCK_SIZE_H)[:, None] == batch & tl.arange(BLOCK_SIZE_W)[None, :] == channel, idx, max_idx)

    for batch in range(BLOCK_SIZE_H):
        for channel in range(BLOCK_SIZE_W):
            n = off_n + batch
            c = off_c + channel
            if n < input_n and c < input_c:
                out_offset = n * input_c * output_h * output_w + c * output_h * output_w + pid_out_h * output_w + pid_out_w
                tl.store(output_ptr + out_offset, max_val[batch, channel])
                if return_indices:
                    tl.store(indices_ptr + out_offset, max_idx[batch, channel])

def fused_fractional_max_pool2d_with_relu(
    input: torch.Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices=False
) -> torch.Tensor:
    assert input.dim() == 4, "Input must be 4D (N, C, H, W)"
    if (output_size is None) == (output_ratio is None):
        raise ValueError("Exactly one of output_size or output_ratio must be specified")
    
    N, C, H_in, W_in = input.shape
    if output_ratio is not None:
        H_out = int(H_in * output_ratio[0])
        W_out = int(W_in * output_ratio[1])
    else:
        H_out, W_out = output_size
    
    if isinstance(kernel_size, int):
        kernel_h = kernel_size
        kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size
    
    stride_h_scale = (H_in - kernel_h) / (H_out - 1) if H_out > 1 else 0.0
    stride_w_scale = (W_in - kernel_w) / (W_out - 1) if W_out > 1 else 0.0

    output = torch.empty((N, C, H_out, W_out), device=input.device, dtype=input.dtype)
    indices = torch.empty((N, C, H_out, W_out), device=input.device, dtype=torch.long) if return_indices else None

    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    grid = (triton.cdiv(N * C, BLOCK_SIZE_H * BLOCK_SIZE_W) * H_out * W_out,)
    
    fused_relu_fractional_max_pool2d_kernel[grid](
        input,
        output,
        indices,
        N, C, H_in, W_in,
        kernel_h, kernel_w,
        H_out, W_out,
        stride_h_scale, stride_w_scale,
        return_indices,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return (output, indices) if return_indices else output
