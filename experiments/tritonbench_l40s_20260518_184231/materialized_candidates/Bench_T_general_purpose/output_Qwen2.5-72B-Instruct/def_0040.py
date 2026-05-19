import triton
import triton.language as tl

@triton.jit
def batch_norm_sigmoid_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    output_ptr, N, C, L, training, momentum, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input, running mean, and running var
    input = tl.load(input_ptr + offsets, mask=offsets < N * C * L, other=0.0)
    running_mean = tl.load(running_mean_ptr + offsets % C, mask=offsets < N * C * L, other=0.0)
    running_var = tl.load(running_var_ptr + offsets % C, mask=offsets < N * C * L, other=0.0)

    # Compute normalized input
    normalized = (input - running_mean) / tl.sqrt(running_var + eps)

    # Apply weight and bias if provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + offsets % C, mask=offsets < N * C * L, other=1.0)
        normalized *= weight
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets % C, mask=offsets < N * C * L, other=0.0)
        normalized += bias

    # Apply sigmoid activation
    output = 1.0 / (1.0 + tl.exp(-normalized))

    # Store the output
    tl.store(output_ptr + offsets, output, mask=offsets < N * C * L)

import torch
import triton
import triton.language as tl

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Ensure input is a tensor
    assert isinstance(input, torch.Tensor), "input must be a torch.Tensor"
    assert input.dim() in [2, 3], "input must have 2 or 3 dimensions (N, C) or (N, C, L)"

    # Ensure running_mean and running_var are tensors
    assert isinstance(running_mean, torch.Tensor), "running_mean must be a torch.Tensor"
    assert isinstance(running_var, torch.Tensor), "running_var must be a torch.Tensor"

    # Ensure weight and bias are tensors if provided
    if weight is not None:
        assert isinstance(weight, torch.Tensor), "weight must be a torch.Tensor"
    if bias is not None:
        assert isinstance(bias, torch.Tensor), "bias must be a torch.Tensor"

    # Get input shape
    N, C = input.shape[:2]
    L = input.shape[2] if input.dim() == 3 else 1

    # Ensure running_mean and running_var have the correct shape
    assert running_mean.shape == (C,), "running_mean must have shape (C,)"
    assert running_var.shape == (C,), "running_var must have shape (C,)"

    # Ensure weight and bias have the correct shape if provided
    if weight is not None:
        assert weight.shape == (C,), "weight must have shape (C,)"
    if bias is not None:
        assert bias.shape == (C,), "bias must have shape (C,)"

    # Allocate output tensor
    output = torch.empty_like(input)

    # Launch the Triton kernel
    grid = (triton.cdiv(N * C * L, 1024),)
    batch_norm_sigmoid_kernel[grid](
        input, running_mean, running_var, weight, bias,
        output, N, C, L, training, momentum, eps,
        BLOCK_SIZE=1024
    )

    return output
