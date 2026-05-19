import triton
import triton.language as tl
import numpy as np

@triton.jit
def adaptive_avg_pool2d_kernel(input, output, N, C, H_in, W_in, H_out, W_out):
    """
    Kernel for adaptive average pooling.

    Args:
        input: Input tensor of shape (N, C, H_in, W_in) or (C, H_in, W_in).
        output: Output tensor of shape (N, C, H_out, W_out) or (C, H_out, W_out).
        N: Batch size (only used if input is (N, C, H_in, W_in)).
        C: Number of input planes.
        H_in: Height of input.
        W_in: Width of input.
        H_out: Target output height.
        W_out: Target output width.
    """
    # Calculate the output indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    h_out = tl.program_id(2)
    w_out = tl.program_id(3)

    # Calculate the pooling region
    h_start = h_out * H_in // H_out
    h_end = (h_out + 1) * H_in // H_out
    w_start = w_out * W_in // W_out
    w_end = (w_out + 1) * W_in // W_out

    # Compute the average
    sum_val = 0.0
    count = 0
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            if h < H_in and w < W_in:
                sum_val += input[batch_idx, channel_idx, h, w]
                count += 1

    output[batch_idx, channel_idx, h_out, w_out] = sum_val / count if count > 0 else 0.0

def adaptive_avg_pool2d(output_size):
    """
    Wrapper function for adaptive average pooling.

    Args:
        output_size: Target output size (single integer or double-integer tuple).

    Returns:
        Tensor: Output tensor after applying adaptive average pooling.
    """
    # Determine output dimensions
    if isinstance(output_size, int):
        H_out = W_out = output_size
    elif isinstance(output_size, tuple) and len(output_size) == 2:
        H_out, W_out = output_size
    else:
        raise ValueError("output_size must be an int or a tuple of two ints.")

    # Input shape (N, C, H_in, W_in) or (C, H_in, W_in)
    input_shape = ...  # Define input shape based on your use case
    output_shape = (input_shape[0], input_shape[1], H_out, W_out) if len(input_shape) == 4 else (input_shape[0], H_out, W_out)

    # Allocate output tensor
    output = np.empty(output_shape, dtype=np.float32)

    # Launch the kernel
    grid = (input_shape[0], input_shape[1], H_out, W_out) if len(input_shape) == 4 else (input_shape[0], H_out, W_out)
    adaptive_avg_pool2d_kernel[grid](input, output, *input_shape, H_out, W_out)

    return output
