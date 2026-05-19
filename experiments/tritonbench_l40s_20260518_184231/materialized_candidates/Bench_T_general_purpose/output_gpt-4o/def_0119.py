import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    stride_h, stride_w, padding_h, padding_w,
    dilation_h, dilation_w, groups,
    in_channels, out_channels, kernel_h, kernel_w,
    in_h, in_w, out_h, out_w,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the block index and thread index
    bh = tl.program_id(0)
    bw = tl.program_id(1)
    
    # Calculate the starting point of the output block
    oh = bh * BLOCK_SIZE
    ow = bw * BLOCK_SIZE
    
    # Loop over the block
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            # Calculate the output position
            h_out = oh + i
            w_out = ow + j
            
            # Initialize the output value
            out_val = 0.0
            
            # Loop over the input channels and kernel size
            for c in range(in_channels):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        # Calculate the input position
                        h_in = h_out * stride_h - padding_h + kh * dilation_h
                        w_in = w_out * stride_w - padding_w + kw * dilation_w
                        
                        # Check if the input position is within bounds
                        if 0 <= h_in < in_h and 0 <= w_in < in_w:
                            # Load the input and weight values
                            input_val = tl.load(input_ptr + (c * in_h + h_in) * in_w + w_in)
                            weight_val = tl.load(weight_ptr + ((c * kernel_h + kh) * kernel_w + kw))
                            
                            # Accumulate the convolution result
                            out_val += input_val * weight_val
            
            # Add the bias if provided
            if bias_ptr:
                out_val += tl.load(bias_ptr + h_out * out_w + w_out)
            
            # Store the result
            tl.store(output_ptr + h_out * out_w + w_out, out_val)

def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
    # Extract dimensions
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1
    
    # Prepare output tensor
    output = torch.empty((batch_size, out_channels, out_h, out_w), device=input.device, dtype=input.dtype)
    
    # Launch the Triton kernel
    BLOCK_SIZE = 16
    grid = (triton.cdiv(out_h, BLOCK_SIZE), triton.cdiv(out_w, BLOCK_SIZE))
    conv2d_kernel[grid](
        input, weight, bias, output,
        stride, stride, padding, padding,
        dilation, dilation, groups,
        in_channels, out_channels, kernel_h, kernel_w,
        in_h, in_w, out_h, out_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Perform pixel shuffle
    upscale_out_h = out_h * upscale_factor
    upscale_out_w = out_w * upscale_factor
    upscale_output = output.view(batch_size, upscale_factor, upscale_factor, out_channels // (upscale_factor ** 2), out_h, out_w)
    upscale_output = upscale_output.permute(0, 3, 4, 1, 5, 2).contiguous()
    upscale_output = upscale_output.view(batch_size, out_channels // (upscale_factor ** 2), upscale_out_h, upscale_out_w)
    
    return upscale_output
