import triton
import triton.language as tl

@triton.jit
def adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    H_in, W_in, H_out, W_out,
    stride_n, stride_c, stride_h, stride_w,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(2)
    hw_out = pid
    h_out = hw_out // W_out
    w_out = hw_out % W_out
    n = tl.program_id(0)
    c = tl.program_id(1)

    start_h = h_out * H_in // H_out
    end_h = (h_out + 1) * H_in // H_out
    start_w = w_out * W_in // W_out
    end_w = (w_out + 1) * W_in // W_out

    total = 0.0
    count = 0
    for h_in in range(start_h, end_h):
        for w_in in range(start_w, end_w):
            input_idx = n * stride_n + c * stride_c + h_in * stride_h + w_in * stride_w
            val = tl.load(input_ptr + input_idx)
            total += val
            count += 1
    avg = total / count if count != 0 else 0.0
    output_idx = n * stride_on + c * stride_oc + h_out * stride_oh + w_out * stride_ow
    tl.store(output_ptr + output_idx, avg)

def adaptive_avg_pool2d(input: torch.Tensor, output_size) -> torch.Tensor:
    if input.dim() not in [3, 4]:
        raise ValueError("Input must be 3D or 4D")
    is_3d = input.dim() == 3
    if is_3d:
        input = input.unsqueeze(0)
    N, C, H_in, W_in = input.shape
    if isinstance(output_size, int):
        H_out = W_out = output_size
    elif isinstance(output_size, tuple):
        H_out = output_size[0] if output_size[0] is not None else H_in
        W_out = output_size[1] if output_size[1] is not None else W_in
    else:
        raise ValueError("output_size must be int or tuple")
    H_out, W_out = int(H_out), int(W_out)
    output = torch.empty((N, C, H_out, W_out), device=input.device, dtype=input.dtype)
    grid = (N, C, H_out * W_out)
    adaptive_avg_pool2d_kernel[grid](
        input, output,
        H_in, W_in, H_out, W_out,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=16
    )
    return output.squeeze(0) if is_3d else output
