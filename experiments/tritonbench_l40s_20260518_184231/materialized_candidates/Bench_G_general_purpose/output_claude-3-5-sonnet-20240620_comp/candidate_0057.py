import torch
import triton
import triton.language as tl

# Define block sizes for parallelization
BLOCK_SIZE_M = 128  # For batch and output features
BLOCK_SIZE_N = 32   # For output height and width
BLOCK_SIZE_K = 64   # For input features and kernel dimensions

@triton.jit
def conv2d_forward_kernel(
    # Pointers to input arrays
    input_ptr, weight_ptr, output_ptr,
    # Dimensions
    batch, in_channels, out_channels,
    in_height, in_width,
    out_height, out_width,
    kernel_h, kernel_w,
    # Strides for data layout
    stride_h, stride_w,
    padding_h, padding_w,
    # Array strides
    input_batch_stride, input_channel_stride, input_height_stride,
    weight_output_stride, weight_input_stride, weight_height_stride,
    output_batch_stride, output_channel_stride, output_height_stride,
    # Optional configurations
    groups: tl.constexpr,
    USE_FP16: tl.constexpr,
    USE_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Calculate indices for this thread block
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(batch * out_channels, BLOCK_M)
    num_pid_n = tl.cdiv(out_height * out_width, BLOCK_N)
    
    # Calculate batch and feature indices
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Calculate batch and channel indices
    batch_idx = offs_m // out_channels
    out_channel_idx = offs_m % out_channels
    
    # Calculate spatial indices
    out_height_idx = offs_n // out_width
    out_width_idx = offs_n % out_width
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main convolution loop
    for k in range(0, in_channels // groups):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input spatial positions
                in_h = out_height_idx * stride_h + kh - padding_h
                in_w = out_width_idx * stride_w + kw - padding_w
                
                # Load input values
                in_mask = (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width)
                input_offs = (batch_idx * input_batch_stride +
                            k * input_channel_stride +
                            in_h * input_height_stride +
                            in_w)
                in_vals = tl.load(input_ptr + input_offs, mask=in_mask, other=0.0)
                
                # Load weight values
                weight_offs = (out_channel_idx * weight_output_stride +
                             k * weight_input_stride +
                             kh * weight_height_stride +
                             kw)
                weight_vals = tl.load(weight_ptr + weight_offs)
                
                # Accumulate
                acc += tl.dot(in_vals, weight_vals)
    
    # Write output
    output_offs = (batch_idx * output_batch_stride +
                  out_channel_idx * output_channel_stride +
                  out_height_idx * output_height_stride +
                  out_width_idx)
    
    output_mask = (batch_idx < batch) & (out_channel_idx < out_channels)
    tl.store(output_ptr + output_offs, acc, mask=output_mask)

def conv2d_forward(input_tensor, weight_tensor, stride=(1, 1), padding=(0, 0), groups=1):
    # Extract dimensions
    batch, in_channels, in_height, in_width = input_tensor.shape
    out_channels, in_channels_per_group, kernel_h, kernel_w = weight_tensor.shape
    
    # Compute output dimensions
    out_height = ((in_height + 2 * padding[0] - kernel_h) // stride[0]) + 1
    out_width = ((in_width + 2 * padding[1] - kernel_w) // stride[1]) + 1
    
    # Initialize output tensor
    output = torch.empty((batch, out_channels, out_height, out_width),
                        device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Calculate grid and block sizes
    grid = lambda meta: (
        triton.cdiv(batch * out_channels, BLOCK_SIZE_M) *
        triton.cdiv(out_height * out_width, BLOCK_SIZE_N),
    )
    
    # Launch kernel
    conv2d_forward_kernel[grid](
        input_tensor, weight_tensor, output,
        batch, in_channels, out_channels,
        in_height, in_width,
        out_height, out_width,
        kernel_h, kernel_w,
        stride[0], stride[1],
        padding[0], padding[1],
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2),
        weight_tensor.stride(0), weight_tensor.stride(1), weight_tensor.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        groups=groups,
        USE_FP16=(input_tensor.dtype == torch.float16),
        USE_TF32=True,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_N=BLOCK_SIZE_N,
        BLOCK_K=BLOCK_SIZE_K,
    )
    
    return output
