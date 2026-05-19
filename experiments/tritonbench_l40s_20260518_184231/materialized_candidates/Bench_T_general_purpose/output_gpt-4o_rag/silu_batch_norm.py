import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def silu_batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_channels,
    eps,
    training,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate program ID and offsets
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    # Load the input, running mean, and running variance
    input = tl.load(input_ptr + pid * n_channels + offsets, mask=offsets < n_channels)
    running_mean = tl.load(running_mean_ptr + offsets, mask=offsets < n_channels)
    running_var = tl.load(running_var_ptr + offsets, mask=offsets < n_channels)

    # Perform batch normalization
    normalized = (input - running_mean) / tl.sqrt(running_var + eps)

    # Apply scale (weight) and shift (bias) if provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + offsets, mask=offsets < n_channels)
        normalized *= weight

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets, mask=offsets < n_channels)
        normalized += bias

    # Apply SiLU activation function
    silu = normalized * tl.sigmoid(normalized)

    # Store the result
    tl.store(output_ptr + pid * n_channels + offsets, silu, mask=offsets < n_channels)

@torch.inference_mode()
def silu_batch_norm(input: Tensor, running_mean: Tensor, running_var: Tensor, weight: Tensor = None, bias: Tensor = None, training: bool = False, momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    """
    Applies Batch Normalization followed by SiLU activation to the input tensor.

    Args:
        input (Tensor): The input tensor for Batch Normalization.
        running_mean (Tensor): The running mean tensor (used during evaluation).
        running_var (Tensor): The running variance tensor (used during evaluation).
        weight (Tensor, optional): The weight tensor for Batch Normalization scaling. Default: None.
        bias (Tensor, optional): The bias tensor for Batch Normalization. Default: None.
        training (bool, optional): Whether the module is in training mode. Default: False.
        momentum (float, optional): Value used for the running mean and variance computation. Default: 0.1.
        eps (float, optional): A small value added to the denominator for numerical stability. Default: 1e-5.

    Returns:
        Tensor: The output tensor after applying Batch Normalization and SiLU activation.
    """
    n_channels = input.size(-1)
    output = torch.empty_like(input)

    # Define grid size
    grid = (input.numel() // n_channels,)

    # Launch kernel
    silu_batch_norm_kernel[grid](
        input,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        n_channels,
        eps,
        training,
        BLOCK_SIZE=triton.next_power_of_2(n_channels),
        num_warps=4,
        num_stages=2
    )

    return output
