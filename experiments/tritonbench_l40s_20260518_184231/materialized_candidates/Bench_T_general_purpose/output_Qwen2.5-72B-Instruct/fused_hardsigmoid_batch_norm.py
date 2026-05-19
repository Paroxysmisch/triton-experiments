import triton
import triton.language as tl

@triton.jit
def batch_norm_hardsigmoid_kernel(
    X,  # Input tensor
    Y,  # Output tensor
    running_mean,  # Running mean
    running_var,  # Running variance
    weight,  # Weight (optional)
    bias,  # Bias (optional)
    stride,  # Stride for the batch dimension
    N,  # Number of elements in the batch
    C,  # Number of channels
    H,  # Height
    W,  # Width
    eps,  # Small constant for numerical stability
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    # Compute the block index and the starting point in the batch
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the range of elements this block will process
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load the input elements
    x = tl.load(X + offsets, mask=offsets < N * C * H * W, other=0.0)

    # Compute the channel index
    c = (offsets % (C * H * W)) // (H * W)

    # Load the running mean and variance
    mean = tl.load(running_mean + c)
    var = tl.load(running_var + c)

    # Apply batch normalization
    normalized_x = (x - mean) * tl.rsqrt(var + eps)

    # Apply weight and bias if provided
    if weight is not None:
        w = tl.load(weight + c)
        normalized_x *= w
    if bias is not None:
        b = tl.load(bias + c)
        normalized_x += b

    # Apply Hardsigmoid activation
    y = tl.where(normalized_x <= -3.0, 0.0, tl.where(normalized_x >= 3.0, 1.0, normalized_x / 6.0 + 0.5))

    # Store the output
    tl.store(Y + offsets, y, mask=offsets < N * C * H * W)

import torch
import triton
import triton.language as tl

def fused_hardsigmoid_batch_norm(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
    inplace: bool = False
) -> torch.Tensor:
    # Ensure the input tensor is on the same device as the running mean and variance
    assert x.device == running_mean.device == running_var.device, "Input tensor and running statistics must be on the same device."

    # Ensure the input tensor is a 4D tensor (N, C, H, W)
    assert x.dim() == 4, "Input tensor must be 4D (N, C, H, W)."

    # Get the shape of the input tensor
    N, C, H, W = x.shape

    # Allocate the output tensor
    if inplace:
        y = x
    else:
        y = torch.empty_like(x)

    # Convert the PyTorch tensors to Triton tensors
    x_triton = x.contiguous().view(-1)
    y_triton = y.contiguous().view(-1)
    running_mean_triton = running_mean.contiguous()
    running_var_triton = running_var.contiguous()
    weight_triton = weight.contiguous() if weight is not None else None
    bias_triton = bias.contiguous() if bias is not None else None

    # Define the grid and block size
    grid = (N * C * H * W + 1024 - 1) // 1024
    block_size = 1024

    # Call the Triton kernel
    batch_norm_hardsigmoid_kernel[grid, block_size](
        x_triton, y_triton, running_mean_triton, running_var_triton, weight_triton, bias_triton,
        N * C * H * W, N, C, H, W, eps, BLOCK_SIZE=block_size
    )

    # Update the running mean and variance if in training mode
    if training:
        # Compute the mean and variance of the input tensor
        mean = x.mean(dim=[0, 2, 3])
        var = x.var(dim=[0, 2, 3], unbiased=False)

        # Update the running mean and variance
        running_mean.mul_(1 - momentum).add_(mean * momentum)
        running_var.mul_(1 - momentum).add_(var * momentum)

    return y
