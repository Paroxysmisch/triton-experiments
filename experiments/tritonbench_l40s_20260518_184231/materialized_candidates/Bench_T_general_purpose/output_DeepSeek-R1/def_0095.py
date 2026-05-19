import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm_kernel(
    input_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, output_ptr,
    eps, C, M,
    has_weight: tl.constexpr, has_bias: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr
):
    pid_c = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    if pid_c >= C or pid_m * BLOCK_SIZE_M >= M:
        return
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask = offs_m < M
    
    input_ptrs = input_ptr + pid_c * M + offs_m
    x = tl.load(input_ptrs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + pid_c)
    var = tl.load(var_ptr + pid_c)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    normalized = (x - mean) * inv_std
    
    if has_weight:
        weight = tl.load(weight_ptr + pid_c)
        normalized *= weight
    if has_bias:
        bias = tl.load(bias_ptr + pid_c)
        normalized += bias
    
    output_ptrs = output_ptr + pid_c * M + offs_m
    tl.store(output_ptrs, normalized, mask=mask)

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    assert input.dim() >= 2, "Input must have at least 2 dimensions"
    shape = input.shape
    C = shape[1]
    input_2d = input.transpose(0, 1).contiguous().view(C, -1)
    M = input_2d.shape[1]
    
    if training:
        mean = input_2d.mean(dim=1)
        var = input_2d.var(dim=1, unbiased=False)
        with torch.no_grad():
            running_mean.copy_((1 - momentum) * running_mean + momentum * mean)
            running_var.copy_((1 - momentum) * running_var + momentum * var)
    else:
        mean = running_mean
        var = running_var
    
    output = torch.empty_like(input_2d)
    has_weight = weight is not None
    has_bias = bias is not None
    weight_ptr = weight.data_ptr() if has_weight else 0
    bias_ptr = bias.data_ptr() if has_bias else 0
    
    BLOCK_SIZE_M = 128
    grid = (C, triton.cdiv(M, BLOCK_SIZE_M))
    
    batch_norm_kernel[grid](
        input_2d.data_ptr(), mean.data_ptr(), var.data_ptr(),
        weight_ptr, bias_ptr, output.data_ptr(),
        eps, C, M, has_weight, has_bias,
        BLOCK_SIZE_M=BLOCK_SIZE_M
    )
    
    output = output.view(shape[1], shape[0], *shape[2:]).transpose(0, 1).contiguous()
    return output
