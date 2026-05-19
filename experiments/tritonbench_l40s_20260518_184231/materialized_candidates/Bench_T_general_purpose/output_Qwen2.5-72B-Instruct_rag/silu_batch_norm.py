import torch
import triton
import triton.language as tl

# Triton kernel for batch normalization followed by SiLU activation
@triton.jit
def silu_batch_norm_kernel(
    input_ptr,
    output_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    input_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Load input, running mean, and running var
    input_block = tl.load(input_ptr + pid * input_row_stride + offsets, mask=mask)
    running_mean = tl.load(running_mean_ptr + offsets, mask=mask)
    running_var = tl.load(running_var_ptr + offsets, mask=mask)

    # Batch normalization
    normalized = (input_block - running_mean) * tl.rsqrt(running_var + eps)

    # Apply weight and bias if provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + offsets, mask=mask)
        normalized = normalized * weight
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets, mask=mask)
        normalized = normalized + bias

    # SiLU activation
    sigmoid = 1 / (1 + tl.exp(-normalized))
    output_block = normalized * sigmoid

    # Store the result
    tl.store(output_ptr + pid * input_row_stride + offsets, output_block, mask=mask)

# Wrapper function to set up and launch the kernel
@torch.inference_mode()
def silu_batch_norm(input: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None, training: bool = False, momentum: float = 0.1, eps: float = 1e-5) -> torch.Tensor:
    """
    Applies Batch Normalization over an input tensor across channels, followed by the Sigmoid Linear Unit (SiLU) activation function applied element-wise.

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
    if training:
        # Update running mean and variance (not implemented in this example)
        pass

    n_cols = input.shape[-1]
    input_row_stride = input.stride(-2)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(input)

    grid = (input.numel() // n_cols,)

    silu_batch_norm_kernel[grid](
        input,
        output,
        running_mean,
        running_var,
        weight if weight is not None else triton.language.core.empty((0,), dtype=triton.language.core.float32),
        bias if bias is not None else triton.language.core.empty((0,), dtype=triton.language.core.float32),
        input_row_stride,
        n_cols,
        eps,
        BLOCK_SIZE,
        num_warps=4,
        num_stages=2
    )

    return output
