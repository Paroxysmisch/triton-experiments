import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_adaptive_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    H,
    W,
    output_h,
    output_w,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    output_stride_n,
    output_stride_c,
    output_stride_h,
    output_stride_w,
    N,
    C,
):
    pid_nc = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)
    
    n = pid_nc // C
    c = pid_nc % C
    
    start_h = (oh * H) // output_h
    end_h = ((oh + 1) * H + output_h - 1) // output_h
    start_w = (ow * W) // output_w
    end_w = ((ow + 1) * W + output_w - 1) // output_w
    
    window_size = (end_h - start_h) * (end_w - start_w)
    total = 0.0
    
    for h in range(start_h, end_h):
        for w in range(start_w, end_w):
            input_offset = n * stride_n + c * stride_c + h * stride_h + w * stride_w
            total += tl.load(input_ptr + input_offset)
    
    avg = total / window_size
    sigmoid_avg = 1.0 / (1.0 + tl.exp(-avg))
    
    output_offset = n * output_stride_n + c * output_stride_c + oh * output_stride_h + ow * output_stride_w
    tl.store(output_ptr + output_offset, sigmoid_avg)

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: int or tuple) -> torch.Tensor:
    assert input.dim() == 4, "Input must be a 4D tensor (N, C, H, W)"
    N, C, H, W = input.shape
    
    if isinstance(output_size, int):
        output_h = output_w = output_size
    else:
        output_h, output_w = output_size
    
    output = torch.empty((N, C, output_h, output_w), device=input.device, dtype=input.dtype)
    
    stride_n, stride_c, stride_h, stride_w = input.stride()
    output_stride_n, output_stride_c, output_stride_h, output_stride_w = output.stride()
    
    grid = (N * C, output_h, output_w)
    
    sigmoid_adaptive_avg_pool2d_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        H=H,
        W=W,
        output_h=output_h,
        output_w=output_w,
        stride_n=stride_n,
        stride_c=stride_c,
        stride_h=stride_h,
        stride_w=stride_w,
        output_stride_n=output_stride_n,
        output_stride_c=output_stride_c,
        output_stride_h=output_stride_h,
        output_stride_w=output_stride_w,
        N=N,
        C=C,
    )
    
    return output
