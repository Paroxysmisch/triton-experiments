import triton
import triton.language as tl
import torch

@triton.jit
def conv2d_forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    input_height, input_width, input_channels,
    output_height, output_width, output_channels,
    kernel_height, kernel_width,
    stride_height, stride_width,
    padding_height, padding_width,
    input_stride_h, input_stride_w, input_stride_c,
    weight_stride_o, weight_stride_c, weight_stride_h, weight_stride_w,
    output_stride_h, output_stride_w, output_stride_c,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    # Calculate the block index
    bh = tl.program_id(0)
    bw = tl.program_id(1)
    
    # Calculate the starting position for this block
    oh = bh * BLOCK_SIZE_H
    ow = bw * BLOCK_SIZE_W

    # Iterate over the output feature map for this block
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            for oc in range(output_channels):
                # Calculate the input position
                ih = oh * stride_height + kh - padding_height
                iw = ow * stride_width + kw - padding_width

                # Check if input position is valid
                if 0 <= ih < input_height and 0 <= iw < input_width:
                    for ic in range(input_channels):
                        # Load input and weight values
                        input_val = tl.load(input_ptr + ih * input_stride_h + iw * input_stride_w + ic * input_stride_c)
                        weight_val = tl.load(weight_ptr + oc * weight_stride_o + ic * weight_stride_c + kh * weight_stride_h + kw * weight_stride_w)
                        
                        # Compute and accumulate the result
                        output_val = tl.load(output_ptr + oh * output_stride_h + ow * output_stride_w + oc * output_stride_c)
                        output_val += input_val * weight_val
                        tl.store(output_ptr + oh * output_stride_h + ow * output_stride_w + oc * output_stride_c, output_val)

def conv2d_forward(input: torch.Tensor, weight: torch.Tensor, stride=(1, 1), padding=(0, 0)):
    # Extract dimensions
    input_height, input_width, input_channels = input.shape
    output_channels, _, kernel_height, kernel_width = weight.shape
    stride_height, stride_width = stride
    padding_height, padding_width = padding

    # Calculate output dimensions
    output_height = (input_height + 2 * padding_height - kernel_height) // stride_height + 1
    output_width = (input_width + 2 * padding_width - kernel_width) // stride_width + 1

    # Initialize output tensor
    output = torch.zeros((output_height, output_width, output_channels), device=input.device, dtype=input.dtype)

    # Define block sizes
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8

    # Launch the Triton kernel
    grid = (triton.cdiv(output_height, BLOCK_SIZE_H), triton.cdiv(output_width, BLOCK_SIZE_W))
    conv2d_forward_kernel[grid](
        input, weight, output,
        input_height, input_width, input_channels,
        output_height, output_width, output_channels,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_height, padding_width,
        input.stride(0), input.stride(1), input.stride(2),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W
    )

    return output
