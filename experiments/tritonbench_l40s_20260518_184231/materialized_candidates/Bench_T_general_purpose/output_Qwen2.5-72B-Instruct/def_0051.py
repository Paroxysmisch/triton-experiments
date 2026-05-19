import triton
import triton.language as tl

@triton.jit
def cos_avg_pool1d_kernel(
    input_ptr, output_ptr, input_shape, kernel_size, stride, padding, ceil_mode, count_include_pad,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the output shape
    minibatch, in_channels, iW = input_shape
    oW = (iW + 2 * padding - kernel_size + stride - 1) // stride + 1
    if ceil_mode and (iW + 2 * padding - kernel_size) % stride != 0:
        oW += 1

    # Get the program ID
    pid = tl.program_id(axis=0)
    if pid >= minibatch * in_channels * oW:
        return

    # Compute the indices
    minibatch_id = pid // (in_channels * oW)
    channel_id = (pid % (in_channels * oW)) // oW
    output_id = pid % oW

    # Compute the input indices
    input_start = output_id * stride - padding
    input_end = input_start + kernel_size

    # Initialize the sum and count
    sum_val = tl.zeros((1,), dtype=tl.float32)
    count = 0

    # Compute the cosine and average
    for i in range(input_start, input_end):
        if 0 <= i < iW:
            input_val = tl.load(input_ptr + minibatch_id * in_channels * iW + channel_id * iW + i)
            cos_val = tl.cos(input_val)
            sum_val += cos_val
            count += 1
        elif count_include_pad:
            sum_val += 0.0
            count += 1

    # Compute the average
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0

    # Store the result
    tl.store(output_ptr + minibatch_id * in_channels * oW + channel_id * oW + output_id, avg_val)

import torch
import triton
import triton.language as tl

def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> torch.Tensor:
    # Default stride to kernel_size if not provided
    if stride is None:
        stride = kernel_size

    # Get the input shape
    minibatch, in_channels, iW = input.shape

    # Compute the output shape
    oW = (iW + 2 * padding - kernel_size + stride - 1) // stride + 1
    if ceil_mode and (iW + 2 * padding - kernel_size) % stride != 0:
        oW += 1

    # Create the output tensor
    output = torch.empty((minibatch, in_channels, oW), device=input.device, dtype=input.dtype)

    # Define the grid and block sizes
    grid = (minibatch * in_channels * oW,)
    block = (1,)

    # Launch the kernel
    cos_avg_pool1d_kernel[grid, block](
        input, output, (minibatch, in_channels, iW), kernel_size, stride, padding, ceil_mode, count_include_pad,
        BLOCK_SIZE=1
    )

    return output

# Example usage
input_tensor = torch.randn(2, 3, 10)
output_tensor = cos_avg_pool1d(input_tensor, kernel_size=2, stride=1, padding=1, ceil_mode=True, count_include_pad=False)

print(output_tensor)
