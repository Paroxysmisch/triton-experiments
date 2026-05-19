import triton
import triton.language as tl

@triton.jit
def adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr, input_shape, output_shape,
    stride_h, stride_w, block_size: tl.constexpr
):
    pid = tl.program_id(axis=0)
    n, c, h_in, w_in = input_shape
    h_out, w_out = output_shape

    # Compute the block indices
    block_h = pid // w_out
    block_w = pid % w_out

    # Compute the starting and ending indices for the block
    start_h = block_h * stride_h
    start_w = block_w * stride_w
    end_h = min(start_h + stride_h, h_in)
    end_w = min(start_w + stride_w, w_in)

    # Initialize the sum and count for the block
    sum = tl.zeros((block_size, block_size), dtype=tl.float32)
    count = 0

    # Iterate over the block
    for i in range(start_h, end_h):
        for j in range(start_w, end_w):
            input_offset = (pid * c * h_in * w_in) + (i * w_in + j)
            sum += tl.load(input_ptr + input_offset)
            count += 1

    # Compute the average
    avg = sum / count

    # Store the result
    output_offset = (pid * c * h_out * w_out) + (block_h * w_out + block_w)
    tl.store(output_ptr + output_offset, avg)

import torch
import triton
import triton.language as tl

def adaptive_avg_pool2d(input: torch.Tensor, output_size) -> torch.Tensor:
    # Determine the output size
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    elif isinstance(output_size, tuple) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])
    elif output_size is None:
        output_size = (input.shape[-2], input.shape[-1])
    
    # Ensure the input is a 4D tensor (N, C, H_in, W_in)
    if input.dim() == 3:
        input = input.unsqueeze(0)
    
    N, C, H_in, W_in = input.shape
    H_out, W_out = output_size

    # Compute the stride for the pooling
    stride_h = H_in // H_out
    stride_w = W_in // W_out

    # Allocate the output tensor
    output = torch.empty((N, C, H_out, W_out), device=input.device, dtype=input.dtype)

    # Define the grid and block sizes
    grid = (H_out * W_out, 1, 1)
    block = (1, 1, 1)

    # Launch the kernel
    adaptive_avg_pool2d_kernel[grid, block](
        input, output, (N, C, H_in, W_in), (H_out, W_out),
        stride_h, stride_w, 1
    )

    # Squeeze the batch dimension if the input was 3D
    if input.dim() == 3:
        output = output.squeeze(0)

    return output
