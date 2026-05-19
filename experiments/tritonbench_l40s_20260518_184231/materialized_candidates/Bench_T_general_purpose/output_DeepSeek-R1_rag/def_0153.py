import torch
import triton
import triton.language as tl

@triton.jit
def adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    H_in, W_in, H_out, W_out,
    stride_n, stride_c, stride_h, stride_w,
    output_stride_n, output_stride_c, output_stride_h, output_stride_w,
    N, C,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    hw = tl.program_id(1)
    
    nc = pid
    n = nc // C
    c = nc % C
    
    h_out = hw // W_out
    w_out = hw % W_out
    
    start_h = h_out * H_in // H_out
    end_h = (h_out + 1) * H_in // H_out
    end_h = tl.where(start_h == end_h, end_h + 1, end_h)
    end_h = tl.minimum(end_h, H_in)
    
    start_w = w_out * W_in // W_out
    end_w = (w_out + 1) * W_in // W_out
    end_w = tl.where(start_w == end_w, end_w + 1, end_w)
    end_w = tl.minimum(end_w, W_in)
    
    total = 0.0
    count = 0
    
    for h in range(start_h, end_h):
        for w in range(start_w, end_w):
            input_offset = n * stride_n + c * stride_c + h * stride_h + w * stride_w
            input_val = tl.load(input_ptr + input_offset)
            total += input_val.to(tl.float32)
            count += 1
    
    avg = total / count if count != 0 else 0.0
    avg = avg.to(tl.load(input_ptr + 0).dtype)
    
    output_offset = n * output_stride_n + c * output_stride_c + h_out * output_stride_h + w_out * output_stride_w
    tl.store(output_ptr + output_offset, avg)

def adaptive_avg_pool2d(input: torch.Tensor, output_size) -> torch.Tensor:
    def _pair(x):
        if isinstance(x, int):
            return (x, x)
        elif isinstance(x, tuple):
            if len(x) == 1:
                return (x[0], x[0])
            else:
                return (x[0] if x[0] is not None else input.size(-2), x[1] if x[1] is not None else input.size(-1))
        elif x is None:
            return (input.size(-2), input.size(-1))
        else:
            raise ValueError("output_size must be a single integer or a tuple of two integers")
    
    output_size = _pair(output_size)
    H_out, W_out = output_size
    
    is_3d = input.dim() == 3
    if is_3d:
        input = input.unsqueeze(0)
    
    N, C, H_in, W_in = input.shape
    
    output = torch.empty((N, C, H_out, W_out), dtype=input.dtype, device=input.device)
    
    grid = (N * C, H_out * W_out)
    
    adaptive_avg_pool2d_kernel[grid](
        input, output,
        H_in, W_in, H_out, W_out,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        N, C,
        BLOCK_SIZE=1
    )
    
    if is_3d:
        output = output.squeeze(0)
    
    return output
