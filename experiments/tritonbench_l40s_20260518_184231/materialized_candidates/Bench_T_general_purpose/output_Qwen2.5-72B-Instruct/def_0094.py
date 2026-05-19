import triton
import triton.language as tl

@triton.jit
def dropout_sigmoid_linear_kernel(
    X, W, B, Y, dropout_mask, p, scale, stride_x, stride_w, stride_b, stride_y, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_offsets = tl.arange(0, BLOCK_SIZE) % stride_x
    w_offsets = tl.arange(0, BLOCK_SIZE) % stride_w

    # Load input and weight
    x = tl.load(X + x_offsets, mask=mask, other=0.0)
    w = tl.load(W + w_offsets, mask=mask, other=0.0)

    # Linear transformation
    linear = tl.dot(x, w)

    # Add bias if provided
    if B is not None:
        b = tl.load(B, mask=mask, other=0.0)
        linear += b

    # Sigmoid activation
    sigmoid = 1 / (1 + tl.exp(-linear))

    # Dropout
    if p > 0.0:
        dropout_mask = tl.load(dropout_mask + offsets, mask=mask, other=0.0)
        sigmoid = tl.where(dropout_mask, sigmoid * scale, 0.0)

    # Store the result
    tl.store(Y + offsets, sigmoid, mask=mask)

import torch
import triton
import triton.language as tl

def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
    # Check input dimensions
    assert input.dim() >= 2, "Input tensor must have at least 2 dimensions"
    assert weight.dim() == 2, "Weight tensor must be 2D"
    assert bias is None or bias.dim() == 1, "Bias tensor must be 1D"

    # Get input and output shapes
    in_features = input.size(-1)
    out_features = weight.size(0)
    batch_size = input.numel() // in_features

    # Create output tensor
    if inplace:
        output = input
    else:
        output = torch.empty((batch_size, out_features), device=input.device, dtype=input.dtype)

    # Create dropout mask if training
    if training and p > 0.0:
        dropout_mask = torch.rand((batch_size, out_features), device=input.device) < p
        scale = 1 / (1 - p)
    else:
        dropout_mask = None
        scale = 1.0

    # Triton grid and block sizes
    BLOCK_SIZE = 1024
    grid = (output.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch Triton kernel
    dropout_sigmoid_linear_kernel[grid, BLOCK_SIZE](
        input, weight, bias, output, dropout_mask, p, scale,
        in_features, in_features, out_features, out_features, output.numel(), BLOCK_SIZE
    )

    return output
