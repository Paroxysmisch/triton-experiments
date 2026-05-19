import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def _sigmoid_batch_norm(
    input,
    running_mean,
    running_var,
    weight,
    bias,
    training,
    momentum,
    eps,
    output,
    input_mean,
    input_var,
    BLOCK_SIZE: tl.constexpr,
):
    # Triton kernel for applying sigmoid batch normalization
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < input.shape[1]

    # Compute mean and variance
    inp = tl.load(input + row_idx * input.shape[1] + col_offsets, mask, other=0.0).to(tl.float32)
    x_mean = tl.sum(inp) / input.shape[1]
    x_zm = inp - x_mean
    x_var = tl.sum(x_zm * x_zm) / input.shape[1]

    # Update running statistics
    if training:
        alpha = momentum
        r_mean = tl.load(running_mean + col_offsets, mask, other=0.0)
        r_var = tl.load(running_var + col_offsets, mask, other=0.0)
        new_mean = alpha * r_mean + (1 - alpha) * x_mean
        tl.store(running_mean + col_offsets, new_mean, mask)
        new_var = alpha * r_var + (1 - alpha) * x_var
        tl.store(running_var + col_offsets, new_var, mask)
    else:
        new_mean = tl.load(running_mean + col_offsets, mask, other=0.0)
        new_var = tl.load(running_var + col_offsets, mask, other=0.0)

    # Normalize and apply linear transformation
    tl.store(input_mean + row_idx, new_mean, mask=row_idx < input.shape[0])
    tl.store(input_var + row_idx, new_var, mask=row_idx < input.shape[0])
    x_hat = x_zm / tl.sqrt(new_var + eps)
    w = tl.load(weight + col_offsets, mask, other=0.0)
    b = tl.load(bias + col_offsets, mask, other=0.0)
    x_bar = x_hat * w + b

    # Apply sigmoid
    sigmoid = lambda x: 1 / (1 + tl.exp(-x))
    out = sigmoid(x_bar)

    # Write output
    tl.store(output + row_idx * input.shape[1] + col_offsets, out, mask)

def sigmoid_batch_norm(
    input: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Tensor = None,
    bias: Tensor = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    input_mean = torch.empty_like(running_mean)
    input_var = torch.empty_like(running_var)
    output = torch.empty_like(input)
    N, C = input.shape

    # Ensure tensors are contiguous
    if input.stride(0) > 1 and input.stride(1) > 1:
        input = input.contiguous()
    if running_mean.stride(0) > 1:
        running_mean = running_mean.contiguous()
    if running_var.stride(0) > 1:
        running_var = running_var.contiguous()
    if weight is not None and weight.stride(0) > 1:
        weight = weight.contiguous()
    if bias is not None and bias.stride(0) > 1:
        bias = bias.contiguous()
    if output.stride(0) > 1 and output.stride(1) > 1:
        output = output.contiguous()

    grid = lambda META: (N, C)
    _sigmoid_batch_norm[grid](
        input,
        running_mean,
        running_var,
        weight,
        bias,
        training,
        momentum,
        eps,
        output,
        input_mean,
        input_var,
    )
    return output
