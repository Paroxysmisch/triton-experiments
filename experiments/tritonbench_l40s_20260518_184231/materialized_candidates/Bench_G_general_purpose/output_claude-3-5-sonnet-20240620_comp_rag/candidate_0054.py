import triton
import triton.language as tl
import torch

@triton.jit
def conv2d_forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch, in_channels, out_channels, in_height, in_width, out_height, out_width,
    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    weight_out_channel_stride, weight_in_channel_stride, weight_height_stride, weight_width_stride,
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_IN_C: tl.constexpr, BLOCK_SIZE_OUT_C: tl.constexpr,
    USE_FP16: tl.constexpr, USE_TF32: tl.constexpr
):
    # Compute the output position
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_out_h = tl.program_id(2)
    
    # Compute the input indices
    out_w_start = pid_out_h * BLOCK_SIZE_OUT_C
    out_h = pid_out_h
    
    # Iterate over the output width in blocks
    for out_w in range(out_w_start, out_w_start + BLOCK_SIZE_OUT_C):
        if out_w < out_width:
            # Compute the corresponding input position
            in_h_start = out_h * stride_h - padding_h
            in_w_start = out_w * stride_w - padding_w
            
            # Initialize accumulator
            acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT_C), dtype=tl.float32)
            
            # Convolution loop
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    for in_c in range(0, in_channels, BLOCK_SIZE_IN_C):
                        # Load input
                        in_h = in_h_start + kh
                        in_w = in_w_start + kw
                        
                        i_ptr = input_ptr + (
                            pid_batch * input_batch_stride +
                            in_c * input_channel_stride +
                            in_h * input_height_stride +
                            in_w * input_width_stride
                        )
                        
                        # Load weight
                        w_ptr = weight_ptr + (
                            pid_out_c * weight_out_channel_stride +
                            in_c * weight_in_channel_stride +
                            kh * weight_height_stride +
                            kw * weight_width_stride
                        )
                        
                        # Perform matrix multiplication
                        i = tl.load(i_ptr, mask=(in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width), other=0.0)
                        w = tl.load(w_ptr)
                        
                        if USE_FP16:
                            i = tl.float16(i)
                            w = tl.float16(w)
                        elif USE_TF32:
                            i = tl.tf32(i)
                            w = tl.tf32(w)
                        
                        acc += tl.dot(i, w)
            
            # Store output
            o_ptr = output_ptr + (
                pid_batch * output_batch_stride +
                pid_out_c * output_channel_stride +
                out_h * output_height_stride +
                out_w * output_width_stride
            )
            
            tl.store(o_ptr, acc, mask=(out_w < out_width))

def conv2d_forward(input, weight, stride, padding, groups=1, use_fp16=False, use_tf32=False):
    # Extract dimensions
    batch, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Compute output dimensions
    out_height = (in_height + 2 * padding[0] - kernel_h) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - kernel_w) // stride[1] + 1
    
    # Initialize output tensor
    output = torch.empty((batch, out_channels, out_height, out_width), device=input.device, dtype=input.dtype)
    
    # Compute strides
    input_strides = input.stride()
    weight_strides = weight.stride()
    output_strides = output.stride()
    
    # Define block sizes
    BLOCK_SIZE_BATCH = 1
    BLOCK_SIZE_IN_C = 32
    BLOCK_SIZE_OUT_C = 32
    
    # Launch kernel
    grid = (batch, out_channels, out_height)
    conv2d_forward_kernel[grid](
        input, weight, output,
        batch, in_channels, out_channels, in_height, in_width, out_height, out_width,
        kernel_h, kernel_w, stride[0], stride[1], padding[0], padding[1],
        input_strides[0], input_strides[1], input_strides[2], input_strides[3],
        weight_strides[0], weight_strides[1], weight_strides[2], weight_strides[3],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3],
        BLOCK_SIZE_BATCH, BLOCK_SIZE_IN_C, BLOCK_SIZE_OUT_C,
        use_fp16, use_tf32
    )
    
    return output
