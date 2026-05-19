import torch
import triton
import triton.language as tl

@triton.jit
def avg_pool1d_triton_kernel(
    input_ptr,
    output_ptr,
    input_width,
    kernel_size,
    stride,
    padding,
    count_include_pad,
    output_width,
    input_batch_stride,
    input_channel_stride,
    input_width_stride,
    output_batch_stride,
    output_channel_stride,
    output_width_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_ow = tl.program_id(2)
    
    if pid_b >= input_batch_stride or pid_c >= input_channel_stride or pid_ow >= output_width:
        return
    
    window_start = pid_ow * stride - padding
    sum_val = 0.0
    count = 0
    
    for k in range(kernel_size):
        pos = window_start + k
        if pos >= 0 and pos < input_width:
            input_offset = pid_b * input_batch_stride + pid_c * input_channel_stride + pos * input_width_stride
            val = tl.load(input_ptr + input_offset)
            sum_val += val
        if count_include_pad or (pos >= 0 and pos < input_width):
            count += 1
    
    avg = sum_val / count if count != 0 else 0.0
    output_offset = pid_b * output_batch_stride + pid_c * output_channel_stride + pid_ow * output_width_stride
    tl.store(output_ptr + output_offset, avg)

def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> torch.Tensor:
    cos_input = torch.cos(input)
    batch_size, in_channels, input_width = cos_input.shape
    
    if stride is None:
        stride = kernel_size
    
    numerator = input_width + 2 * padding - kernel_size
    if ceil_mode:
        numerator += stride - 1
    output_width = numerator // stride + 1
    output_width = max(output_width, 1)
    
    output = torch.empty((batch_size, in_channels, output_width), device=cos_input.device, dtype=cos_input.dtype)
    
    if kernel_size <= 0:
        raise ValueError("kernel_size must be greater than zero")
    
    grid = (batch_size, in_channels, output_width)
    
    avg_pool1d_triton_kernel[grid](
        cos_input, output,
        input_width,
        kernel_size,
        stride,
        padding,
        count_include_pad,
        output_width,
        cos_input.stride(0),
        cos_input.stride(1),
        cos_input.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_SIZE=1,
    )
    
    return output
