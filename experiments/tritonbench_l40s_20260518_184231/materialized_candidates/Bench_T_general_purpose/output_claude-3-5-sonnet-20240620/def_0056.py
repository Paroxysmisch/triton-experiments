import torch
import triton
import triton.language as tl

@triton.jit
def fused_fractional_maxpool2d_relu_kernel(
    # Pointers to input/output tensors
    input_ptr, output_ptr, indices_ptr,
    # Tensor dimensions
    batch_size, channels, in_height, in_width,
    out_height, out_width,
    # Pooling parameters 
    kh, kw,
    # Strides and other parameters
    input_batch_stride, input_channel_stride, input_row_stride, input_col_stride,
    output_batch_stride, output_channel_stride, output_row_stride, output_col_stride,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate batch and channel indices
    batch_idx = pid // (channels * out_height * out_width)
    remaining = pid % (channels * out_height * out_width)
    channel_idx = remaining // (out_height * out_width)
    out_y = (remaining // out_width) % out_height
    out_x = remaining % out_width

    # Calculate input region boundaries
    start_h = (out_y * in_height) // out_height
    end_h = ((out_y + 1) * in_height) // out_height
    start_w = (out_x * in_width) // out_width
    end_w = ((out_x + 1) * in_width) // out_width

    # Initialize max value and index
    max_val = float('-inf')
    max_idx = 0

    # Input offset for current batch and channel
    input_offset = (batch_idx * input_batch_stride + 
                   channel_idx * input_channel_stride)

    # Pooling window loop
    for h in range(start_h, end_h):
        for w in range(start_w, end_w):
            # Load input value
            idx = input_offset + h * input_row_stride + w * input_col_stride
            val = tl.load(input_ptr + idx)
            
            # Apply ReLU
            val = tl.maximum(0.0, val)
            
            # Update max if necessary
            if val > max_val:
                max_val = val
                max_idx = idx

    # Calculate output offset
    output_offset = (batch_idx * output_batch_stride +
                    channel_idx * output_channel_stride +
                    out_y * output_row_stride +
                    out_x * output_col_stride)

    # Store output and indices
    tl.store(output_ptr + output_offset, max_val)
    if indices_ptr:
        tl.store(indices_ptr + output_offset, max_idx)

def fused_fractional_max_pool2d_with_relu(
    input: torch.Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices=False
) -> torch.Tensor:
    # Input validation
    if not isinstance(kernel_size, (int, tuple)):
        raise TypeError("kernel_size must be int or tuple")
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
        
    if output_size is None and output_ratio is None:
        raise ValueError("Either output_size or output_ratio must be specified")
        
    if output_size is not None and output_ratio is not None:
        raise ValueError("Only one of output_size or output_ratio can be specified")

    # Calculate output dimensions
    batch_size, channels, in_height, in_width = input.shape
    
    if output_ratio is not None:
        out_height = int(in_height * output_ratio[0])
        out_width = int(in_width * output_ratio[1])
    else:
        out_height, out_width = output_size

    # Prepare output tensor and indices if needed
    output = torch.empty((batch_size, channels, out_height, out_width),
                        device=input.device, dtype=input.dtype)
    indices = torch.empty_like(output, dtype=torch.int64) if return_indices else None

    # Calculate grid size
    grid = (batch_size * channels * out_height * out_width,)
    
    # Launch kernel
    fused_fractional_maxpool2d_relu_kernel[grid](
        input, output, indices,
        batch_size, channels, in_height, in_width,
        out_height, out_width,
        kernel_size[0], kernel_size[1],
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=32,
    )

    if return_indices:
        return output, indices
    return output
