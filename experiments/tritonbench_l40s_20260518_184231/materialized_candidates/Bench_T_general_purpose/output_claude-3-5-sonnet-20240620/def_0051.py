import torch
import triton
import triton.language as tl
import math

@triton.jit
def cos_avg_pool1d_kernel(
    # Pointers to input and output tensors
    input_ptr, output_ptr,
    # Tensor dimensions and strides
    batch_size, channels, input_width,
    input_stride_b, input_stride_c, input_stride_w,
    output_stride_b, output_stride_c, output_stride_w,
    # Pooling parameters
    kernel_size, stride, padding,
    # Other parameters
    count_include_pad: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output dimensions
    output_width = ((input_width + 2 * padding - kernel_size) // stride + 1)
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate batch and channel indices
    batch_idx = pid // (channels * output_width)
    tmp = pid % (channels * output_width)
    channel_idx = tmp // output_width
    width_idx = tmp % output_width
    
    # Calculate input start position
    in_w_start = width_idx * stride - padding
    
    # Initialize accumulator
    acc = 0.0
    count = 0
    
    # Perform pooling
    for k in range(kernel_size):
        in_w = in_w_start + k
        if 0 <= in_w < input_width:
            # Calculate input offset
            in_offset = (batch_idx * input_stride_b + 
                        channel_idx * input_stride_c + 
                        in_w * input_stride_w)
            
            # Load input value and compute cosine
            x = tl.load(input_ptr + in_offset)
            cos_x = tl.cos(x)
            
            acc += cos_x
            count += 1
        elif count_include_pad:
            count += 1
    
    # Compute average
    if count > 0:
        acc = acc / float(count)
    
    # Calculate output offset
    out_offset = (batch_idx * output_stride_b + 
                 channel_idx * output_stride_c + 
                 width_idx * output_stride_w)
    
    # Store result
    tl.store(output_ptr + out_offset, acc)

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
        padding (int, optional): Zero-padding added to both sides. Default: 0
        ceil_mode (bool, optional): Use ceil instead of floor for output shape. Default: False
        count_include_pad (bool, optional): Include padding in average calculation. Default: True
    
    Returns:
        Tensor: Output tensor
    """
    # Input validation
    assert input.dim() == 3, "Input must be 3D tensor (batch, channels, width)"
    
    # Default stride to kernel_size if not specified
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
    output = torch.empty(
        (batch_size, channels, output_width),
        device=input.device,
        dtype=input.dtype
    )
    
    # Get tensor strides
    input_stride_b, input_stride_c, input_stride_w = input.stride()
    output_stride_b, output_stride_c, output_stride_w = output.stride()
    
    # Launch kernel
    grid = (batch_size * channels * output_width,)
    cos_avg_pool1d_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        input_width=input_width,
        input_stride_b=input_stride_b,
        input_stride_c=input_stride_c,
        input_stride_w=input_stride_w,
        output_stride_b=output_stride_b,
        output_stride_c=output_stride_c,
        output_stride_w=output_stride_w,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        count_include_pad=count_include_pad,
        BLOCK_SIZE=128,
    )
    
    return output
