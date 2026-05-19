import triton
import triton.language as tl
import torch
import math

@triton.jit
def conv2d_forward_kernel(
    # Pointers to tensors
    input_ptr, weight_ptr, output_ptr,
    # Input dimensions
    batch_size, in_channels, in_height, in_width,
    # Weight dimensions  
    out_channels, kernel_height, kernel_width,
    # Output dimensions
    out_height, out_width,
    # Strides and padding
    stride_h, stride_w, padding_h, padding_w,
    # Tensor strides
    input_strides, weight_strides, output_strides,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate output position
    n = pid // (out_channels * out_height * out_width)
    rem = pid % (out_channels * out_height * out_width)
    oc = rem // (out_height * out_width) 
    h = (rem // out_width) % out_height
    w = rem % out_width

    # Calculate input window position
    h_in = h * stride_h - padding_h
    w_in = w * stride_w - padding_w

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # Iterate over input channels and kernel dimensions
    for ic in range(in_channels):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                h_loc = h_in + kh
                w_loc = w_in + kw
                
                # Check boundaries
                if 0 <= h_loc < in_height and 0 <= w_loc < in_width:
                    # Load input value
                    input_idx = (n * input_strides[0] + 
                               ic * input_strides[1] +
                               h_loc * input_strides[2] + 
                               w_loc * input_strides[3])
                    input_val = tl.load(input_ptr + input_idx)
                    
                    # Load weight value
                    weight_idx = (oc * weight_strides[0] +
                                ic * weight_strides[1] +
                                kh * weight_strides[2] +
                                kw * weight_strides[3])
                    weight_val = tl.load(weight_ptr + weight_idx)
                    
                    # Accumulate product
                    acc += input_val * weight_val

    # Store output
    output_idx = (n * output_strides[0] +
                 oc * output_strides[1] +
                 h * output_strides[2] +
                 w * output_strides[3])
    tl.store(output_ptr + output_idx, acc)

def conv2d_forward(input_tensor, weight_tensor, stride=(1, 1), padding=(0, 0)):
    # Get dimensions
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight_tensor.shape
    
    # Calculate output dimensions
    out_height = ((in_height + 2 * padding[0] - kernel_height) // stride[0]) + 1
    out_width = ((in_width + 2 * padding[1] - kernel_width) // stride[1]) + 1
    
    # Create output tensor
    output = torch.zeros((batch_size, out_channels, out_height, out_width), 
                        device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Get tensor strides
    input_strides = input_tensor.stride()
    weight_strides = weight_tensor.stride()
    output_strides = output.stride()
    
    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # Calculate grid size
    grid = (batch_size * out_channels * out_height * out_width,)
    
    # Launch kernel
    conv2d_forward_kernel[grid](
        input_tensor, weight_tensor, output,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_height, kernel_width,
        out_height, out_width,
        stride[0], stride[1], padding[0], padding[1],
        input_strides, weight_strides, output_strides,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return output
