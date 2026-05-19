import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm_hardsigmoid_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C, eps, inplace: tl.constexpr, training: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * C + tl.arange(0, C)
    
    x = tl.load(x_ptr + offsets)
    mean = tl.load(mean_ptr + offsets)
    var = tl.load(var_ptr + offsets)
    
    # Batch normalization
    inv_std = tl.rsqrt(var + eps)
    x_norm = (x - mean) * inv_std
    
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + offsets)
        x_norm *= weight
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets)
        x_norm += bias
    
    # Hardsigmoid activation
    if inplace:
        x_norm = tl.maximum(0, tl.minimum(1, x_norm * 0.2 + 0.5))
        tl.store(x_ptr + offsets, x_norm)
    else:
        x_hardsigmoid = tl.maximum(0, tl.minimum(1, x_norm * 0.2 + 0.5))
        tl.store(output_ptr + offsets, x_hardsigmoid)

def fused_hardsigmoid_batch_norm(
    x: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor,
    weight: torch.Tensor = None, bias: torch.Tensor = None,
    training: bool = False, momentum: float = 0.1, eps: float = 1e-5,
    inplace: bool = False
) -> torch.Tensor:
    assert x.ndim == 2, "Input tensor must be 2D"
    N, C = x.shape

    if training:
        # Update running mean and variance if in training mode
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
        running_var.mul_(1 - momentum).add_(batch_var, alpha=momentum)
    else:
        batch_mean = running_mean
        batch_var = running_var

    output = x if inplace else torch.empty_like(x)
    
    grid = (N,)
    batch_norm_hardsigmoid_kernel[grid](
        x, batch_mean, batch_var, weight, bias, output,
        N, C, eps, inplace, training
    )

    return output
