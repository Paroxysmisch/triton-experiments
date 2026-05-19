import torch
import triton
import triton.language as tl

@triton.jit
def cos_avg_pool1d_kernel(
    input_ptr, output_ptr,
    BATCH, CHANNELS, WIDTH,
    OUT_WIDTH, KERNEL_SIZE, STRIDE, PADDING,
    CEIL_MODE, COUNT_INCLUDE_PAD,
    BLOCK_SIZE: tl.constexpr
):
    # Define the block indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Compute the start and end positions of the pooling window
    out_pos = tl.arange(0, BLOCK_SIZE)
    start = out_pos * STRIDE - PADDING
    end = start + KERNEL_SIZE

    # Clamp start and end positions
    start = tl.max(0, start)
    end = tl.min(WIDTH, end)

    # Compute input offsets
    input_offset = batch_idx * CHANNELS * WIDTH + channel_idx * WIDTH
    input_ptr = input_ptr + input_offset

    # Compute output offset
    output_offset = batch_idx * CHANNELS * OUT_WIDTH + channel_idx * OUT_WIDTH
    output_ptr = output_ptr + output_offset

    # Initialize output
    output = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the pooling window
    for i in range(KERNEL_SIZE):
        in_pos = start + i
        valid = (in_pos < end)
        in_pos = tl.where(valid, in_pos, 0)
        
        # Load the input, apply cosine, and accumulate
        input_val = tl.load(input_ptr + in_pos, mask=valid, other=0.0)
        output += tl.cos(input_val) * valid.to(tl.float32)

    # Compute the pooling factor
    pool_factor = KERNEL_SIZE if COUNT_INCLUDE_PAD else (end - start).to(tl.float32)
    pool_factor = tl.where(pool_factor == 0, 1, pool_factor)  # Avoid division by zero

    # Compute the average
    output /= pool_factor

    # Store the result
    tl.store(output_ptr + out_pos, output)


def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> torch.Tensor:
    if stride is None:
        stride = kernel_size

    # Get input dimensions
    batch_size, in_channels, iW = input.shape

    # Compute output dimensions
    if ceil_mode:
        oW = (iW + 2 * padding - kernel_size + stride - 1) // stride + 1
    else:
        oW = (iW + 2 * padding - kernel_size) // stride + 1

    # Allocate output tensor
    output = torch.empty((batch_size, in_channels, oW), device=input.device, dtype=input.dtype)

    # Launch Triton kernel
    BLOCK_SIZE = triton.next_power_of_2(oW)
    grid = (batch_size, in_channels)
    cos_avg_pool1d_kernel[grid](
        input, output,
        batch_size, in_channels, iW,
        oW, kernel_size, stride, padding,
        ceil_mode, count_include_pad,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output
