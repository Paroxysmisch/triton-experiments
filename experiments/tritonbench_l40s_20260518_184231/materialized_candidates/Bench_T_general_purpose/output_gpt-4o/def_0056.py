import torch
import triton
import triton.language as tl

@triton.jit
def relu_and_fractional_max_pool2d_kernel(
    input_ptr, output_ptr, indices_ptr,
    input_height, input_width, output_height, output_width,
    kernel_height, kernel_width, stride_height, stride_width,
    return_indices, BLOCK_SIZE: tl.constexpr
):
    # Compute the grid index
    batch_id = tl.program_id(0)
    plane_id = tl.program_id(1)
    out_y = tl.program_id(2)
    out_x = tl.program_id(3)

    # Calculate input starting point for this block
    in_y_start = out_y * stride_height
    in_x_start = out_x * stride_width

    # Initialize max value and index
    max_val = tl.float32(-float('inf'))
    max_index = -1

    # Loop over the pooling window
    for ky in range(kernel_height):
        for kx in range(kernel_width):
            in_y = in_y_start + ky
            in_x = in_x_start + kx
            if in_y < input_height and in_x < input_width:
                idx = batch_id * input_height * input_width + plane_id * input_height * input_width + in_y * input_width + in_x
                val = tl.load(input_ptr + idx)
                # Apply ReLU
                val = tl.max(val, 0.0)
                # Fractional max pooling
                if val > max_val:
                    max_val = val
                    max_index = idx

    # Store the result
    out_idx = batch_id * output_height * output_width + plane_id * output_height * output_width + out_y * output_width + out_x
    tl.store(output_ptr + out_idx, max_val)
    if return_indices:
        tl.store(indices_ptr + out_idx, max_index)

def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
    assert input.dim() == 4, "Input tensor must be 4D (batch, channels, height, width)"
    
    # Determine output size
    if output_size is None:
        if output_ratio is None:
            raise ValueError("Either output_size or output_ratio must be provided")
        output_height = int(input.size(2) * output_ratio[0])
        output_width = int(input.size(3) * output_ratio[1])
    else:
        output_height, output_width = output_size

    # Kernel size
    if isinstance(kernel_size, int):
        kernel_height, kernel_width = kernel_size, kernel_size
    else:
        kernel_height, kernel_width = kernel_size

    # Stride calculation for fractional pooling
    stride_height = (input.size(2) - kernel_height) // (output_height - 1) if output_height > 1 else input.size(2) - kernel_height
    stride_width = (input.size(3) - kernel_width) // (output_width - 1) if output_width > 1 else input.size(3) - kernel_width

    # Allocate output tensor
    output = torch.empty((input.size(0), input.size(1), output_height, output_width), device=input.device, dtype=input.dtype)
    indices = torch.empty_like(output, dtype=torch.int32) if return_indices else None

    # Launch Triton kernel
    grid = (input.size(0), input.size(1), output_height, output_width)
    relu_and_fractional_max_pool2d_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        indices_ptr=indices,
        input_height=input.size(2),
        input_width=input.size(3),
        output_height=output_height,
        output_width=output_width,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        stride_height=stride_height,
        stride_width=stride_width,
        return_indices=return_indices,
        BLOCK_SIZE=1024  # This should be adjusted based on the hardware
    )

    if return_indices:
        return output, indices
    else:
        return output
