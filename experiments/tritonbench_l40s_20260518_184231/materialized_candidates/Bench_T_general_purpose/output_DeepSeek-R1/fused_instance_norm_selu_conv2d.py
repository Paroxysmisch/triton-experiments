import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def fused_selu_instance_norm_kernel(
    input_ptr, output_ptr,
    n_channels, h, w,
    eps,
    gamma_ptr,
    beta_ptr,
    stride_n, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    off_n = pid_n
    off_c = pid_c
    
    input_ptr += off_n * stride_n + off_c * stride_c
    output_ptr += off_n * stride_n + off_c * stride_c
    
    input_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    sum_val = 0.0
    sum_sq = 0.0
    count = 0
    
    for i in range(0, h, BLOCK_SIZE):
        for j in range(0, w, BLOCK_SIZE):
            ih = i + tl.arange(0, BLOCK_SIZE)
            jw = j + tl.arange(0, BLOCK_SIZE)
            mask = (ih < h) & (jw < w)
            
            ptr = input_ptr + ih[:, None] * stride_h + jw[None, :] * stride_w
            elem = tl.load(ptr, mask=mask, other=0.0)
            
            elem_selu = 1.0507009873554804934193349852946 * tl.where(elem >= 0, elem, 1.6732632423543772848170429916717 * (tl.exp(elem) - 1))
            
            input_block = tl.where(mask, elem_selu, input_block)
            
            sum_val += tl.sum(input_block)
            sum_sq += tl.sum(input_block * input_block)
            count += tl.sum(mask)
    
    mean = sum_val / count
    var = (sum_sq / count) - (mean * mean)
    std = tl.sqrt(var + eps)
    
    for i in range(0, h, BLOCK_SIZE):
        for j in range(0, w, BLOCK_SIZE):
            ih = i + tl.arange(0, BLOCK_SIZE)
            jw = j + tl.arange(0, BLOCK_SIZE)
            mask = (ih < h) & (jw < w)
            
            ptr = input_ptr + ih[:, None] * stride_h + jw[None, :] * stride_w
            elem = tl.load(ptr, mask=mask, other=0.0)
            
            elem_selu = 1.0507009873554804934193349852946 * tl.where(elem >= 0, elem, 1.6732632423543772848170429916717 * (tl.exp(elem) - 1))
            
            normalized = (elem_selu - mean) / std
            
            if gamma_ptr is not None:
                gamma = tl.load(gamma_ptr + pid_c)
                beta = tl.load(beta_ptr + pid_c)
                normalized = normalized * gamma + beta
            
            out_ptr = output_ptr + ih[:, None] * stride_h + jw[None, :] * stride_w
            tl.store(out_ptr, normalized, mask=mask)

def fused_instance_norm_selu_conv2d(
    input: torch.Tensor, 
    weight: torch.Tensor, 
    bias: Optional[torch.Tensor] = None, 
    stride=1, 
    padding=0, 
    dilation=1, 
    groups=1, 
    num_features: Optional[int] = None, 
    eps=1e-5, 
    momentum=0.1, 
    affine=False, 
    track_running_stats=False
) -> torch.Tensor:
    # Perform convolution
    conv_out = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    # Determine instance normalization parameters
    out_channels = conv_out.size(1)
    if num_features is None:
        num_features = out_channels
    assert num_features == out_channels, "num_features must match conv output channels"
    
    # Handle affine parameters
    if affine:
        gamma = torch.ones(num_features, device=conv_out.device, dtype=conv_out.dtype)
        beta = torch.zeros(num_features, device=conv_out.device, dtype=conv_out.dtype)
    else:
        gamma, beta = None, None
    
    # Ensure contiguous memory layout
    conv_out = conv_out.contiguous()
    output = torch.empty_like(conv_out)
    
    # Launch Triton kernel
    N, C, H, W = conv_out.shape
    BLOCK_SIZE = 16
    grid = (N, C)
    
    fused_selu_instance_norm_kernel[grid](
        conv_out, output,
        C, H, W,
        eps,
        gamma if affine else None,
        beta if affine else None,
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
