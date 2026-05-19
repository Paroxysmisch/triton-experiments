import triton
import triton.language as tl

@triton.jit
def adaptive_avg_pool2d_kernel(output_ptr, input_ptr, N, C, H_in, W_in, S_0, S_1, stride_h, stride_w, pad_h, pad_w):
    pid = tl.program_id(axis=0)
    block_idx = pid // (S_0 * S_1)
    thread_idx = pid % (S_0 * S_1)

    # Calculate the height and width coordinates for the output pixel
    out_h = thread_idx // S_1
    out_w = thread_idx % S_1

    # Calculate the start and end indices for the input pixels
    in_h_start = max(out_h * stride_h - pad_h, 0)
    in_h_end = min(in_h_start + H_in, H_in)
    in_w_start = max(out_w * stride_w - pad_w, 0)
    in_w_end = min(in_w_start + W_in, W_in)

    # Initialize the sum and count for averaging
    total_sum = tl.zeros([], dtype=tl.float32)
    count = tl.zeros([], dtype=tl.int32)

    # Iterate over the input pixels and accumulate their values
    for h in range(in_h_start, in_h_end):
        for w in range(in_w_start, in_w_end):
            total_sum += input_ptr[block_idx * C * H_in * W_in + (h * W_in + w)]
            count += 1

    # Calculate the average and store the result in the output tensor
    avg_value = total_sum / count
    output_ptr[block_idx * C * S_0 * S_1 + thread_idx] = avg_value

@triton.jit
def adaptive_avg_pool2d(output_size, input_tensor) -> Tensor:
    N, C, H_in, W_in = input_tensor.shape
    if isinstance(output_size, tuple):
        S_0, S_1 = output_size
    else:
        S_0 = S_1 = output_size

    # Calculate padding and stride
    pad_h = ((S_0 - 1) * stride_h + H_in - S_0) // 2
    pad_w = ((S_1 - 1) * stride_w + W_in - S_1) // 2
    stride_h = (H_in + 2 * pad_h - S_0 + stride_h - 1) // stride_h
    stride_w = (W_in + 2 * pad_w - S_1 + stride_w - 1) // stride_w

    # Allocate memory for the output tensor
    output_tensor = tl.zeros((N, C, S_0, S_1), dtype=input_tensor.dtype)

    # Launch the kernel
    grid = (N * C * S_0 * S_1,)
    block = (1,)
    adaptive_avg_pool2d_kernel[grid, block](output_tensor.data, input_tensor.data, N, C, H_in, W_in, S_0, S_1, stride_h, stride_w, pad_h, pad_w)

    return output_tensor
