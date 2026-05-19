import torch
import triton
import triton.language as tl

@triton.jit
def cosine_similarity_kernel(
    x1_ptr, x2_ptr, output_ptr,
    N, C, H, W,
    eps,
    x1_stride_n, x1_stride_c, x1_stride_h, x1_stride_w,
    x2_stride_n, x2_stride_c, x2_stride_h, x2_stride_w,
    output_stride_n, output_stride_h, output_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    
    if n >= N or h >= H or w >= W:
        return
    
    sum_xy = 0.0
    sum_x_sq = 0.0
    sum_y_sq = 0.0
    
    for c in range(0, C, BLOCK_SIZE):
        c_offsets = c + tl.arange(0, BLOCK_SIZE)
        mask = c_offsets < C
        
        x1_ptr_current = x1_ptr + n * x1_stride_n + c_offsets[:, None, None] * x1_stride_c + h * x1_stride_h + w * x1_stride_w
        x2_ptr_current = x2_ptr + n * x2_stride_n + c_offsets[:, None, None] * x2_stride_c + h * x2_stride_h + w * x2_stride_w
        
        x1 = tl.load(x1_ptr_current, mask=mask[:, None, None], other=0.0)
        x2 = tl.load(x2_ptr_current, mask=mask[:, None, None], other=0.0)
        
        sum_xy += tl.sum(x1 * x2)
        sum_x_sq += tl.sum(x1 * x1)
        sum_y_sq += tl.sum(x2 * x2)
    
    norm_x = tl.sqrt(sum_x_sq + eps)
    norm_y = tl.sqrt(sum_y_sq + eps)
    sim = sum_xy / (norm_x * norm_y)
    
    output_ptr_current = output_ptr + n * output_stride_n + h * output_stride_h + w * output_stride_w
    tl.store(output_ptr_current, sim)

def fused_avg_pool2d_cosine_similarity(
    x1: torch.Tensor,
    x2: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    eps: float = 1e-8
) -> torch.Tensor:
    assert x1.dim() == 4 and x2.dim() == 4, "Inputs must be 4D tensors"
    assert x1.shape == x2.shape, "Input shapes must match"
    
    N, C, H, W = x1.shape
    output = torch.empty((N, H, W), device=x1.device, dtype=x1.dtype)
    
    BLOCK_SIZE = 128  # Adjust based on optimal hardware performance
    
    grid = (N, H, W)
    
    x1_ptr = x1.data_ptr()
    x2_ptr = x2.data_ptr()
    output_ptr = output.data_ptr()
    
    x1_stride = x1.stride()
    x2_stride = x2.stride()
    output_stride = output.stride()
    
    cosine_similarity_kernel[grid](
        x1_ptr, x2_ptr, output_ptr,
        N, C, H, W,
        eps,
        x1_stride[0], x1_stride[1], x1_stride[2], x1_stride[3],
        x2_stride[0], x2_stride[1], x2_stride[2], x2_stride[3],
        output_stride[0], output_stride[1], output_stride[2],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    output = output.unsqueeze(1)
    
    if stride is None:
        stride = kernel_size
    output = torch.nn.functional.avg_pool2d(output, kernel_size, stride, padding)
    
    return output
