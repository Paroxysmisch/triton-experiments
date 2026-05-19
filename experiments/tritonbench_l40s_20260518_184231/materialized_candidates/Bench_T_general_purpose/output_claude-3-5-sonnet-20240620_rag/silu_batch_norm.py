import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def batch_norm_kernel(
    input,
    running_mean,
    running_var,
    weight,
    bias,
    output,
    n_cols,
    eps,
    BLOCK_N: tl.constexpr,
):
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    # Load input and parameters
    x = tl.load(input + prog_id * n_cols + offsets, mask=offsets < n_cols)
    mean = tl.load(running_mean)
    var = tl.load(running_var)
    
    # Batch normalization
    x_normalized = (x - mean) / tl.sqrt(var + eps)
    
    # Apply weight and bias if provided
    if weight is not None:
        x_normalized *= tl.load(weight)
    if bias is not None:
        x_normalized += tl.load(bias)

    # SiLU activation
    si = tl.sigmoid(x_normalized)
    out = x_normalized * si

    # Store the result
    tl.store(output + prog_id * n_cols + offsets, out, mask=offsets < n_cols)

@torch.inference_mode()
def silu_batch_norm(
    input: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Tensor = None,
    bias: Tensor = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
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
        Tensor: A tensor that has undergone batch normalization and SiLU activation.
    """

    n_cols = input.shape[1]  # Assuming input is of shape (batch_size, num_features)
    BLOCK_N = triton.next_power_of_2(n_cols)
    output = torch.empty_like(input)

    grid = (input.shape[0],)  # Number of rows in the input tensor
    batch_norm_kernel[grid](
        input,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        n_cols,
        eps,
        BLOCK_N,
        num_warps=4,
        num_stages=2,
        **kernel_meta,
    )

    return output
