import triton
import triton.language as tl

@triton.jit
def batch_norm_silu_kernel(
    X,  # Input tensor
    Y,  # Output tensor
    running_mean,  # Running mean
    running_var,  # Running variance
    weight,  # Weight (optional)
    bias,  # Bias (optional)
    stride,  # Stride for the input tensor
    N,  # Number of elements in the input tensor
    eps,  # Small value for numerical stability
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X + offsets, mask=mask, other=0.0)
    mean = tl.load(running_mean + offsets, mask=mask, other=0.0)
    var = tl.load(running_var + offsets, mask=mask, other=0.0)

    if weight is not None:
        w = tl.load(weight + offsets, mask=mask, other=1.0)
    else:
        w = 1.0

    if bias is not None:
        b = tl.load(bias + offsets, mask=mask, other=0.0)
    else:
        b = 0.0

    # Batch Normalization
    x_normalized = (x - mean) * tl.rsqrt(var + eps) * w + b

    # SiLU Activation
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x_normalized))
    y = x_normalized * sigmoid_x

    tl.store(Y + offsets, y, mask=mask)

import torch
import triton
import triton.language as tl

def silu_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Ensure input is a tensor
    assert isinstance(input, torch.Tensor), "Input must be a torch.Tensor"
    assert isinstance(running_mean, torch.Tensor), "Running mean must be a torch.Tensor"
    assert isinstance(running_var, torch.Tensor), "Running variance must be a torch.Tensor"

    # Ensure optional parameters are tensors if provided
    if weight is not None:
        assert isinstance(weight, torch.Tensor), "Weight must be a torch.Tensor"
    if bias is not None:
        assert isinstance(bias, torch.Tensor), "Bias must be a torch.Tensor"

    # Ensure input and running_mean/running_var have the same number of elements
    assert input.numel() == running_mean.numel() == running_var.numel(), "Input, running mean, and running variance must have the same number of elements"

    # Ensure weight and bias have the same number of elements as input if provided
    if weight is not None:
        assert weight.numel() == input.numel(), "Weight must have the same number of elements as input"
    if bias is not None:
        assert bias.numel() == input.numel(), "Bias must have the same number of elements as input"

    # Create output tensor
    output = torch.empty_like(input)

    # Define grid and block sizes
    BLOCK_SIZE = 1024
    grid = (input.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    batch_norm_silu_kernel[grid, BLOCK_SIZE](
        input, output, running_mean, running_var, weight, bias, input.stride(0), input.numel(), eps
    )

    return output
