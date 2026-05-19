import triton
import triton.language as tl

@triton.jit
def hardsigmoid(x):
    """
    Applies the Hardsigmoid activation function element-wise.
    """
    return tl.where(x > 3, 1, tl.where(x < -3, 0, (x + 3) / 6))

@triton.jit
def batch_norm_and_hardsigmoid_kernel(
    x_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    out_ptr, x_stride, out_stride, C, N, eps, BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for batch normalization followed by Hardsigmoid activation.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x_offsets = offsets * x_stride
    out_offsets = offsets * out_stride

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    # Load running mean and variance
    running_mean = tl.load(running_mean_ptr + tl.arange(0, C)).to(tl.float32)
    running_var = tl.load(running_var_ptr + tl.arange(0, C)).to(tl.float32)

    # Load weight and bias if provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + tl.arange(0, C)).to(tl.float32)
    else:
        weight = tl.full((C,), 1.0, dtype=tl.float32)

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(0, C)).to(tl.float32)
    else:
        bias = tl.full((C,), 0.0, dtype=tl.float32)

    # Batch normalization
    x_normalized = (x - running_mean) * tl.rsqrt(running_var + eps) * weight + bias

    # Hardsigmoid activation
    x_activated = hardsigmoid(x_normalized)

    # Store the result
    tl.store(out_ptr + out_offsets, x_activated, mask=mask)

import torch

def fused_hardsigmoid_batch_norm(
    x: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor,
    weight: torch.Tensor = None, bias: torch.Tensor = None, training: bool = False,
    momentum: float = 0.1, eps: float = 1e-5, inplace: bool = False
) -> torch.Tensor:
    """
    Applies Batch Normalization followed by the Hardsigmoid activation function on the input tensor `x`.
    """
    if training:
        # Update running mean and variance
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * batch_var

    if inplace:
        out = x
    else:
        out = torch.empty_like(x)

    # Launch the Triton kernel
    grid = (x.numel() // 1024 + 1,)
    batch_norm_and_hardsigmoid_kernel[grid](
        x, running_mean, running_var, weight, bias, out,
        x.stride(0), out.stride(0), x.size(1), x.numel(), eps, 1024
    )

    return out
