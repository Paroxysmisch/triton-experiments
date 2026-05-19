import torch
import triton
import triton.language as tl

@triton.jit
def hardsigmoid(x, inplace):
    if inplace:
        x = tl.where(x <= -3.0, 0.0, x)
        x = tl.where(x >= 3.0, 1.0, x)
        x = tl.where((x > -3.0) & (x < 3.0), 0.5 * x + 0.5, x)
    else:
        x = tl.where(x <= -3.0, 0.0, x)
        x = tl.where(x >= 3.0, 1.0, x)
        x = tl.where((x > -3.0) & (x < 3.0), 0.5 * x + 0.5, x)
    return x

@triton.jit
def batch_norm(
    x,
    running_mean,
    running_var,
    weight,
    bias,
    training,
    momentum,
    eps,
    in_place,
    affine,
):
    if training:
        batch_mean = tl.mean(x, axis=0)
        batch_var = tl.var(x, axis=0)
        x_centered = x - batch_mean
        batch_std = tl.sqrt(batch_var + eps)
        x_normed = x_centered / batch_std
        if affine:
            x_out = weight[:, None] * x_normed + bias[:, None]
        else:
            x_out = x_normed
        return x_out, batch_mean, batch_var
    else:
        x_centered = x - running_mean
        batch_std = tl.sqrt(running_var + eps)
        x_normed = x_centered / batch_std
        if affine:
            x_out = weight[:, None] * x_normed + bias[:, None]
        else:
            x_out = x_normed
        return x_out, running_mean, running_var

def fused_hardsigmoid_batch_norm(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
    inplace: bool = False,
) -> torch.Tensor:
    affine = weight is not None and bias is not None
    x_bn, running_mean, running_var = batch_norm(
        x,
        running_mean,
        running_var,
        weight,
        bias,
        training,
        momentum,
        eps,
        inplace,
        affine,
    )
    x_hardsigmoid = hardsigmoid(x_bn, inplace)
    return x_hardsigmoid
