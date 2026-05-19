import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

# Assuming apply_act_func is available and supports 'hardsigmoid'
from .act_kernels import apply_act_func

@triton.jit
def fused_hardsigmoid_batch_norm_kernel(
    x_ptr, running_mean_ptr, running_var_ptr,
    weight_ptr, bias_ptr, output_ptr,
    num_channels, M,
    eps, training, momentum,
    inplace: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    if pid >= num_channels:
        return

    # Compute mean and inverse std for the channel
    if training:
        sum_x = 0.0
        sum_x2 = 0.0
        for i in range(0, M, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            mask = offs < M
            x = tl.load(x_ptr + pid * M + offs, mask=mask, other=0.0)
            sum_x += tl.sum(x)
            sum_x2 += tl.sum(x * x)
        mean = sum_x / M
        var = (sum_x2 / M) - (mean * mean)
        inv_std = 1.0 / tl.sqrt(var + eps)

        # Update running mean and variance
        current_mean = tl.load(running_mean_ptr + pid)
        current_var = tl.load(running_var_ptr + pid)
        updated_mean = (1 - momentum) * current_mean + momentum * mean
        updated_var = (1 - momentum) * current_var + momentum * var
        tl.store(running_mean_ptr + pid, updated_mean)
        tl.store(running_var_ptr + pid, updated_var)
    else:
        mean = tl.load(running_mean_ptr + pid)
        var = tl.load(running_var_ptr + pid)
        inv_std = 1.0 / tl.sqrt(var + eps)

    # Load weight and bias
    weight = 1.0
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + pid)
    bias = 0.0
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid)

    # Standardize and apply Hardsigmoid
    for i in range(0, M, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < M
        x = tl.load(x_ptr + pid * M + offs, mask=mask, other=0.0)
        x_std = (x - mean) * inv_std
        x_scaled = x_std * weight + bias
        hardsigmoid_x = apply_act_func(x_scaled, None, None, None, None, 'hardsigmoid', inplace)
        tl.store(output_ptr + pid * M + offs, hardsigmoid_x, mask=mask)

def fused_hardsigmoid_batch_norm(
    x: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
    inplace: bool = False
) -> Tensor:
    assert x.dim() >= 2, "Input tensor must have at least 2 dimensions"
    C = x.size(1)
    assert running_mean.size(0) == C, "running_mean size mismatch"
    assert running_var.size(0) == C, "running_var size mismatch"
    if weight is not None:
        assert weight.size(0) == C, "Weight size mismatch"
    if bias is not None:
        assert bias.size(0) == C, "Bias size mismatch"

    original_shape = x.shape
    x_2d = x.reshape(C, -1)
    M = x_2d.size(1)

    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    if inplace:
        output = x
        output_2d = x_2d
    else:
        output = torch.empty_like(x)
        output_2d = output.view(C, -1)

    BLOCK_SIZE = 1024
    grid = (C,)
    fused_hardsigmoid_batch_norm_kernel[grid](
        x_2d, running_mean, running_var,
        weight if weight is not None else None,
        bias if bias is not None else None,
        output_2d,
        num_channels=C,
        M=M,
        eps=eps,
        training=training,
        momentum=momentum,
        inplace=inplace,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output.reshape(original_shape)
