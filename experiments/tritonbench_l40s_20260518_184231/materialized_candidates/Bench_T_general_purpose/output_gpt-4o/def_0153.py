import triton
import triton.language as tl

@triton.jit
def adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr, 
    N, C, H_in, W_in, H_out, W_out,
    stride_h, stride_w, kernel_h, kernel_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # Calculate the indices for the current block
    n = tl.program_id(0)
    c = tl.program_id(1)
    h_out = tl.program_id(2) * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out = tl.program_id(3) * BLOCK_W + tl.arange(0, BLOCK_W)

    # Clamp indices to output dimensions
    h_out = tl.where(h_out < H_out, h_out, 0)
    w_out = tl.where(w_out < W_out, w_out, 0)

    # Calculate pooling window
    h_start = h_out * stride_h
    w_start = w_out * stride_w
    h_end = tl.minimum(h_start + kernel_h, H_in)
    w_end = tl.minimum(w_start + kernel_w, W_in)

    # Initialize sum
    sum = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Perform pooling
    for h in range(kernel_h):
        for w in range(kernel_w):
            h_idx = h_start + h
            w_idx = w_start + w
            mask_h = h_idx < h_end
            mask_w = w_idx < w_end
            mask = mask_h[:, None] & mask_w[None, :]
            input_idx = n * C * H_in * W_in + c * H_in * W_in + h_idx * W_in + w_idx
            input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
            sum += input_val

    # Compute average
    pool_size = (h_end - h_start) * (w_end - w_start)
    avg = sum / pool_size

    # Store result
    output_idx = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out
    tl.store(output_ptr + output_idx, avg)

import torch

def adaptive_avg_pool2d(input, output_size):
    # Determine input dimensions
    if input.dim() == 3:
        C, H_in, W_in = input.shape
        N = 1
    elif input.dim() == 4:
        N, C, H_in, W_in = input.shape
    else:
        raise ValueError("Input must be a 3D or 4D tensor")

    # Determine output size
    if isinstance(output_size, int):
        H_out = W_out = output_size
    elif isinstance(output_size, tuple) and len(output_size) == 2:
        H_out, W_out = output_size
    else:
        raise ValueError("Output size must be an int or a tuple of two ints")

    # Handle None values in output_size
    H_out = H_in if H_out is None else H_out
    W_out = W_in if W_out is None else W_out

    # Compute stride and kernel size
    stride_h = H_in // H_out
    stride_w = W_in // W_out
    kernel_h = H_in - (H_out - 1) * stride_h
    kernel_w = W_in - (W_out - 1) * stride_w

    # Prepare output tensor
    output = torch.empty((N, C, H_out, W_out), device=input.device, dtype=input.dtype)

    # Launch Triton kernel
    grid = (N, C, (H_out + 31) // 32, (W_out + 31) // 32)
    adaptive_avg_pool2d_kernel[grid](
        input, output, 
        N, C, H_in, W_in, H_out, W_out,
        stride_h, stride_w, kernel_h, kernel_w,
        BLOCK_H=32, BLOCK_W=32
    )

    # Return output
    if input.dim() == 3:
        return output.squeeze(0)
    return output
