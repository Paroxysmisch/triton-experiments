import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def pixel_shuffle_kernel(
    input_ptr, output_ptr,
    upscale_factor,
    input_c, input_h, input_w,
    output_c, output_h, output_w,
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    output_stride_n, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_rest = tl.program_id(1)
    
    numel_per_batch = output_c * output_h * output_w
    off = pid_rest * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < numel_per_batch
    off = tl.multiple_of(off, BLOCK_SIZE)
    
    c_out = off // (output_h * output_w)
    remainder = off % (output_h * output_w)
    h_out = remainder // output_w
    w_out = remainder % output_w
    
    r = upscale_factor
    r_sq = r * r
    c_in = c_out * r_sq + (h_out % r) * r + (w_out % r)
    h_in = h_out // r
    w_in = w_out // r
    
    input_pos = (pid_n * input_stride_n + c_in * input_stride_c +
                 h_in * input_stride_h + w_in * input_stride_w)
    output_pos = (pid_n * output_stride_n + c_out * output_stride_c +
                  h_out * output_stride_h + w_out * output_stride_w)
    
    input_val = tl.load(input_ptr + input_pos, mask=mask, other=0)
    tl.store(output_ptr + output_pos, input_val, mask=mask)

def pixel_shuffle_triton(input, upscale_factor):
    n, c, h, w = input.shape
    r = upscale_factor
    assert c % (r ** 2) == 0, "input channels must be divisible by upscale_factor squared"
    output_c = c // (r ** 2)
    output_h = h * r
    output_w = w * r
    output = torch.empty((n, output_c, output_h, output_w), dtype=input.dtype, device=input.device)
    
    BLOCK_SIZE = 256
    grid = (n, triton.cdiv(output_c * output_h * output_w, BLOCK_SIZE))
    
    pixel_shuffle_kernel[grid](
        input, output,
        upscale_factor,
        c, h, w,
        output_c, output_h, output_w,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

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
    x = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    return pixel_shuffle_triton(x, upscale_factor)
