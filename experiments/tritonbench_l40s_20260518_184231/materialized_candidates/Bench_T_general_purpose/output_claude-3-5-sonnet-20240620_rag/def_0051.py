import torch
import triton
import triton.language as tl
import math

@triton.jit
def cos_avg_pool1d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_width,
    output_width,
    kernel_size,
    stride,
    padding,
    count_include_pad,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for batch and output position
    pid = tl.program_id(0)
    
    # Calculate batch and channel indices
    batch_idx = pid // (channels * output_width)
    remaining = pid % (channels * output_width)
    channel_idx = remaining // output_width
    out_idx = remaining % output_width
    
    # Calculate input start position
    in_start_idx = out_idx * stride - padding
    
    # Initialize accumulator and counter
    acc = 0.0
    count = 0
    
    # Compute base offset for the current batch and channel
    base_offset = batch_idx * (channels * input_width) + channel_idx * input_width
    
    # Perform pooling operation
    for i in range(kernel_size):
        curr_idx = in_start_idx + i
        if count_include_pad or (curr_idx >= 0 and curr_idx < input_width):
            if curr_idx >= 0 and curr_idx < input_width:
                val = tl.load(input_ptr + base_offset + curr_idx)
                # Apply cosine function
                acc += tl.cos(val)
            count += 1
    
    # Calculate average
    result = acc / count if count > 0 else 0.0
    
    # Store result
    output_offset = batch_idx * (channels * output_width) + channel_idx * output_width + out_idx
    tl.store(output_ptr + output_offset, result)

def cos_avg_pool1d(
    input: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True
) -> torch.Tensor:
    """
    Applies cosine function element-wise followed by 1D average pooling.
    
    Args:
        input (Tensor): Input tensor of shape (minibatch, in_channels, iW)
        kernel_size (int): Size of the pooling window
        stride (int, optional): Stride of the pooling window. Default: kernel_size
        padding (int, optional): Padding added to both sides of the input. Default: 0
        ceil_mode (bool, optional): When True, will use ceil instead of floor to compute output shape. Default: False
        count_include_pad (bool, optional): When True, will include padding in averaging calculation. Default: True
    
    Returns:
        Tensor: Output tensor
    """
    assert input.is_cuda, "Input tensor must be on GPU"
    assert len(input.shape) == 3, "Input must be 3D tensor (batch, channels, width)"
    
    # Set default stride
    if stride is None:
        stride = kernel_size
    
    # Get input dimensions
    batch_size, channels, input_width = input.shape
    
    # Calculate output width
    if ceil_mode:
        output_width = math.ceil((input_width + 2 * padding - kernel_size) / stride + 1)
    else:
        output_width = math.floor((input_width + 2 * padding - kernel_size) / stride + 1)
    
    # Create output tensor
    output = torch.empty((batch_size, channels, output_width), 
                        device=input.device, dtype=input.dtype)
    
    # Calculate grid size
    grid = (batch_size * channels * output_width,)
    
    # Launch kernel
    cos_avg_pool1d_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        input_width=input_width,
        output_width=output_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        count_include_pad=count_include_pad,
        BLOCK_SIZE=32,
    )
    
    return output
