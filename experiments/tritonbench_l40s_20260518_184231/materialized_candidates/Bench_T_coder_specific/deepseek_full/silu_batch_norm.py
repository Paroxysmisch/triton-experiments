import torch
import triton
import triton.language as tl

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def silu(x):
    # SiLU is just a scaled sigmoid
    return x * tl.sigmoid(x)

@triton.jit
def sqrt(x):
    # Fast approximation of the sqrt function
    xhalf = 0.5 * x;
    y = torch.bitcast(x, y);
    i = torch.bitcast(y, i);
    i = 0x5f3759df - (i >> 1);
    y = torch.bitcast(i, y);
    y = y * (1.5 - (xhalf * y * y));
    return y;

@triton.jit
def batch_norm(
    input,
    running_mean,
    running_var,
    weight,
    bias,
    training,
    momentum,
    eps,
):
    # Batch normalization
    return torch.batch_norm(
        input,
        running_mean,
        running_var,
        weight,
        bias,
        training,
        momentum,
        eps,
    )

@triton.jit
def silu_batch_norm(
    input,
    running_mean,
    running_var,
    weight,
    bias,
    training,
    momentum,
    eps,
):
    # Apply batch normalization and silu
    return silu(
        batch_norm(
            input,
            running_mean,
            running_var,
            weight,
            bias,
            training,
            momentum,
            eps,
        )
    )
