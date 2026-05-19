import torch
import triton
import triton.language as tl
import math

@triton.jit
def pixel_shuffle_conv2d_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Matrix dimensions
    batch, in_channels, out_channels, in_height, in_width,
    kernel_height, kernel_width, out_height, out_width,
    # Parameters
    stride, padding, dilation, groups, upscale_factor,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate output position
    out_n = pid // (out_height * out_width)
    out_hw = pid % (out_height * out_width)
    out_h = out_hw // out_width
    out_w = out_hw % out_width
    
    # Calculate input position with stride and padding
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding
    
    # Initialize output accumulator
    acc = tl.zeros((out_channels,), dtype=tl.float32)
    
    # Convolution loop
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            h_in = in_h_start + kh * dilation
            w_in = in_w_start + kw * dilation
            
            if 0 <= h_in < in_height and 0 <= w_in < in_width:
                for ic in range(0, in_channels, BLOCK_SIZE_M):
                    # Load input block
                    input_block = tl.load(
                        input_ptr + (
                            out_n * in_channels * in_height * in_width +
                            ic * in_height * in_width +
                            h_in * in_width + w_in
                        )
                    )
                    
                    # Load weight block
                    for oc in range(0, out_channels, BLOCK_SIZE_N):
                        weight_block = tl.load(
                            weight_ptr + (
                                oc * in_channels * kernel_height * kernel_width +
                                ic * kernel_height * kernel_width +
                                kh * kernel_width + kw
                            )
                        )
                        
                        # Accumulate
                        acc[oc:oc+BLOCK_SIZE_N] += input_block * weight_block

    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(0, out_channels))
        acc += bias
    
    # Pixel shuffle rearrangement
    r = upscale_factor
    oc_out = out_channels // (r * r)
    out_h_ps = out_h * r
    out_w_ps = out_w * r
    
    # Store output with pixel shuffle arrangement
    for oc in range(oc_out):
        for i in range(r):
            for j in range(r):
                out_idx = (
                    out_n * oc_out * (out_height * r) * (out_width * r) +
                    oc * (out_height * r) * (out_width * r) +
                    (out_h_ps + i) * (out_width * r) +
                    (out_w_ps + j)
                )
                channel_idx = oc * r * r + i * r + j
                tl.store(output_ptr + out_idx, acc[channel_idx])

def pixel_shuffle_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    upscale_factor=2
) -> torch.Tensor:
    """
    Applies 2D convolution followed by pixel shuffle upscaling.
    
    Args:
        input: Input tensor of shape (minibatch, in_channels, iH, iW)
        weight: Convolution filter of shape (out_channels, in_channels/groups, kH, kW)
        bias: Optional bias tensor of shape (out_channels)
        stride: Stride of the convolution
        padding: Padding added to all sides of the input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections from input to output channels
        upscale_factor: Factor by which to increase spatial resolution
    
    Returns:
        Output tensor of shape (minibatch, out_channels/(upscale_factor^2),
                              iH*upscale_factor, iW*upscale_factor)
    """
    assert input.is_cuda and weight.is_cuda, "Input and weight must be CUDA tensors"
    if bias is not None:
        assert bias.is_cuda, "Bias must be a CUDA tensor"
    
    # Extract dimensions
    batch, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    
    # Ensure output channels is divisible by upscale_factor squared
    assert out_channels % (upscale_factor * upscale_factor) == 0, \
        "Output channels must be divisible by upscale_factor^2"
    
    # Create output tensor
    output = torch.empty(
        (batch,
         out_channels // (upscale_factor * upscale_factor),
         out_height * upscale_factor,
         out_width * upscale_factor),
        device=input.device,
        dtype=input.dtype
    )
    
    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # Launch kernel
    grid = (batch * out_height * out_width,)
    pixel_shuffle_conv2d_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias if bias is not None else None,
        output_ptr=output,
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        in_height=in_height,
        in_width=in_width,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        out_height=out_height,
        out_width=out_width,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        upscale_factor=upscale_factor,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output
