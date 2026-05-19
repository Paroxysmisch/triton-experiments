import torch
import triton
import triton.language as tl

@triton.jit
def fused_bn_hardsigmoid_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, output_ptr,
    eps, inplace_flag,
    C, channel_stride, num_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    for i in range(pid * BLOCK_SIZE, (pid + 1) * BLOCK_SIZE):
        if i >= num_elements:
            return
        
        c = (i // channel_stride) % C
        
        x_val = tl.load(x_ptr + i)
        mean_val = tl.load(mean_ptr + c)
        var_val = tl.load(var_ptr + c)
        
        normalized = (x_val - mean_val) / tl.sqrt(var_val + eps)
        
        if weight_ptr != 0:
            weight_val = tl.load(weight_ptr + c)
            normalized *= weight_val
        if bias_ptr != 0:
            bias_val = tl.load(bias_ptr + c)
            normalized += bias_val
        
        hardsigmoid_val = normalized * (1.0 / 6.0) + 0.5
        hardsigmoid_val = tl.minimum(tl.maximum(hardsigmoid_val, 0.0), 1.0)
        
        if inplace_flag:
            tl.store(x_ptr + i, hardsigmoid_val)
        else:
            tl.store(output_ptr + i, hardsigmoid_val)

def fused_hardsigmoid_batch_norm(
    x: torch.Tensor, 
    running_mean: torch.Tensor, 
    running_var: torch.Tensor, 
    weight: torch.Tensor = None, 
    bias: torch.Tensor = None, 
    training: bool = False, 
    momentum: float = 0.1, 
    eps: float = 1e-5, 
    inplace: bool = False
) -> torch.Tensor:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert running_mean.is_contiguous() and running_var.is_contiguous(), "Running stats must be contiguous"
    if weight is not None:
        assert weight.is_contiguous(), "Weight must be contiguous"
    if bias is not None:
        assert bias.is_contiguous(), "Bias must be contiguous"
    assert x.size(1) == running_mean.size(0), "Channel dimension mismatch"
    assert x.size(1) == running_var.size(0), "Channel dimension mismatch"
    if weight is not None:
        assert weight.size(0) == x.size(1), "Weight size mismatch"
    if bias is not None:
        assert bias.size(0) == x.size(1), "Bias size mismatch"
    
    if training:
        dims = list(range(x.dim()))
        dims.pop(1)
        batch_mean = x.mean(dim=dims)
        batch_var = x.var(dim=dims, unbiased=False)
        
        with torch.no_grad():
            running_mean.copy_((1 - momentum) * running_mean + momentum * batch_mean)
            running_var.copy_((1 - momentum) * running_var + momentum * batch_var)
        mean = batch_mean
        var = batch_var
    else:
        mean = running_mean
        var = running_var
    
    if inplace:
        output = x
    else:
        output = torch.empty_like(x)
    
    C = x.size(1)
    channel_stride = x.stride(1)
    num_elements = x.numel()
    
    weight_ptr = weight.data_ptr() if weight is not None else 0
    bias_ptr = bias.data_ptr() if bias is not None else 0
    
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    
    fused_bn_hardsigmoid_kernel[grid](
        x.data_ptr(), mean.data_ptr(), var.data_ptr(),
        weight_ptr, bias_ptr,
        output.data_ptr(),
        eps, inplace,
        C, channel_stride, num_elements,
        BLOCK_SIZE=1024,
    )
    
    return output
