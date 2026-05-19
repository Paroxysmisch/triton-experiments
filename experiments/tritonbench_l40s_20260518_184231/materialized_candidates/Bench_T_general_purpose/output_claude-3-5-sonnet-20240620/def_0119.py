import triton
import triton.language as tl
import torch

@triton.jit
def pixel_shuffle_conv2d_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, output_ptr, bias_ptr,
    # Matrix dimensions
    batch, in_channels, in_height, in_width, 
    out_channels, kernel_height, kernel_width,
    # Parameters
    stride, padding, dilation, groups, upscale_factor,
    # Strides for tensors
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    weight_output_stride, weight_input_stride, weight_height_stride, weight_width_stride,
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Calculate output dimensions
    out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    
    # Compute pixel shuffle output dimensions
    shuffle_height = out_height * upscale_factor
    shuffle_width = out_width * upscale_factor
    shuffle_channels = out_channels // (upscale_factor * upscale_factor)
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate batch and output position
    batch_idx = pid // (shuffle_channels * shuffle_height * shuffle_width)
    tmp = pid % (shuffle_channels * shuffle_height * shuffle_width)
    channel_idx = tmp // (shuffle_height * shuffle_width)
    height_idx = (tmp // shuffle_width) % shuffle_height
    width_idx = tmp % shuffle_width
    
    # Load input block
    input_block = tl.load(input_ptr + batch_idx * input_batch_stride + 
                         channel_idx * input_channel_stride +
                         height_idx * input_height_stride +
                         width_idx * input_width_stride)
    
    # Compute convolution
    acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    for k in range(0, in_channels // groups, BLOCK_SIZE_K):
        for h in range(kernel_height):
            for w in range(kernel_width):
                # Load weight block
                weight_block = tl.load(weight_ptr + 
                                     channel_idx * weight_output_stride +
                                     k * weight_input_stride +
                                     h * weight_height_stride +
                                     w * weight_width_stride)
                
                # Compute partial sum
                acc += input_block * weight_block
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + channel_idx)
        acc += bias
    
    # Apply pixel shuffle rearrangement
    shuffle_channel = channel_idx % shuffle_channels
    shuffle_height = height_idx * upscale_factor + (channel_idx // shuffle_channels) // upscale_factor
    shuffle_width = width_idx * upscale_factor + (channel_idx // shuffle_channels) % upscale_factor
    
    # Store output
    output_idx = (batch_idx * output_batch_stride +
                 shuffle_channel * output_channel_stride +
                 shuffle_height * output_height_stride +
                 shuffle_width * output_width_stride)
    tl.store(output_ptr + output_idx, acc)

def pixel_shuffle_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    upscale_factor: int = 2
) -> torch.Tensor:
    assert input.dim() == 4, "Input must be a 4D tensor"
    assert weight.dim() == 4, "Weight must be a 4D tensor"
    
    batch, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    
    # Calculate pixel shuffle dimensions
    shuffle_channels = out_channels // (upscale_factor * upscale_factor)
    shuffle_height = out_height * upscale_factor
    shuffle_width = out_width * upscale_factor
    
    # Create output tensor
    output = torch.empty(
        (batch, shuffle_channels, shuffle_height, shuffle_width),
        device=input.device,
        dtype=input.dtype
    )
    
    # Launch kernel
    grid = lambda meta: (
        batch * shuffle_channels * shuffle_height * shuffle_width,
    )
    
    pixel_shuffle_conv2d_kernel[grid](
        input, weight, output,
        bias if bias is not None else None,
        batch, in_channels, in_height, in_width,
        out_channels, kernel_height, kernel_width,
        stride, padding, dilation, groups, upscale_factor,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
    )
    
    return output
