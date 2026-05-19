import torch
import triton
import triton.language as tl

@triton.jit
def relu_and_pool(input_ptr, output_ptr, indices_ptr, input_height, input_width, output_height, output_width, kernel_height, kernel_width, stride_height, stride_width, return_indices: tl.constexpr):
    # Compute the block indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Compute the starting point of the pooling window
    ih_start = oh * stride_height
    iw_start = ow * stride_width

    # Initialize max value and index
    max_val = -float('inf')
    max_idx = -1

    # Iterate over the pooling window
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            ih = ih_start + kh
            iw = iw_start + kw

            # Check if the indices are within bounds
            if ih < input_height and iw < input_width:
                input_offset = ((batch_idx * input_height + ih) * input_width + iw) * tl.num_programs(1) + channel_idx
                val = tl.load(input_ptr + input_offset)
                # Apply ReLU
                val = tl.max(val, 0.0)

                # Update max value and index
                if val > max_val:
                    max_val = val
                    max_idx = kh * kernel_width + kw

    # Store the max value
    output_offset = ((batch_idx * output_height + oh) * output_width + ow) * tl.num_programs(1) + channel_idx
    tl.store(output_ptr + output_offset, max_val)

    # Optionally store the index
    if return_indices:
        indices_offset = ((batch_idx * output_height + oh) * output_width + ow) * tl.num_programs(1) + channel_idx
        tl.store(indices_ptr + indices_offset, max_idx)

def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
    # Determine the kernel size
    if isinstance(kernel_size, int):
        kernel_height, kernel_width = kernel_size, kernel_size
    else:
        kernel_height, kernel_width = kernel_size

    # Determine the output size
    input_height, input_width = input.shape[-2:]
    if output_size is not None:
        output_height, output_width = output_size
    elif output_ratio is not None:
        output_height = int(input_height * output_ratio[0])
        output_width = int(input_width * output_ratio[1])
    else:
        raise ValueError("Either output_size or output_ratio must be specified")

    # Determine the stride size
    stride_height = (input_height - kernel_height) // (output_height - 1)
    stride_width = (input_width - kernel_width) // (output_width - 1)

    # Allocate output tensor
    output = torch.empty((input.shape[0], input.shape[1], output_height, output_width), device=input.device, dtype=input.dtype)
    indices = torch.empty_like(output, dtype=torch.int32) if return_indices else None

    # Launch Triton kernel
    grid = (input.shape[0], input.shape[1], output_height, output_width)
    relu_and_pool[grid](
        input,
        output,
        indices,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        return_indices
    )

    if return_indices:
        return output, indices
    else:
        return output
