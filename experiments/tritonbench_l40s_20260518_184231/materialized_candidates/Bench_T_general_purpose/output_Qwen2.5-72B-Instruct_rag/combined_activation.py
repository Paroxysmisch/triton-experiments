import torch
import triton
import triton.language as tl

@triton.jit
def combined_activation_kernel(
    X,  # Input tensor
    W1,  # Weight matrix 1
    W2,  # Weight matrix 2
    B,  # Bias tensor
    Y,  # Output tensor
    N,  # Number of elements in the last dimension of X
    D_in,  # Number of elements in the last dimension of X
    D_out,  # Number of elements in the last dimension of Y
    stride_xn,  # Stride for X in the last dimension
    stride_xd,  # Stride for X in the second-to-last dimension
    stride_w1d,  # Stride for W1 in the last dimension
    stride_w2d,  # Stride for W2 in the last dimension
    stride_yd,  # Stride for Y in the last dimension
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load X and W1
    x_offsets = tl.arange(0, D_in) * stride_xd + offsets[:, None] * stride_xn
    x = tl.load(X + x_offsets, mask=mask[:, None], other=0.0)

    w1_offsets = tl.arange(0, D_in) * stride_w1d + tl.arange(0, D_out)[None, :]
    w1 = tl.load(W1 + w1_offsets)

    # Matrix multiplication
    z = tl.dot(x, w1)

    # Sigmoid
    z = 1 / (1 + tl.exp(-z))

    # Tanh
    z = (tl.exp(z) - tl.exp(-z)) / (tl.exp(z) + tl.exp(-z))

    # Load W2 and B
    w2_offsets = offsets[:, None] * stride_w2d + tl.arange(0, D_out)[None, :]
    w2 = tl.load(W2 + w2_offsets, mask=mask[:, None], other=1.0)

    b_offsets = tl.arange(0, D_out)
    b = tl.load(B + b_offsets)

    # Element-wise multiplication and addition
    z = z * w2 + b

    # Store the result
    y_offsets = offsets[:, None] * stride_yd + tl.arange(0, D_out)[None, :]
    tl.store(Y + y_offsets, z, mask=mask[:, None])

import torch

def combined_activation(input, weight1, weight2, bias, *, out=None):
    # Ensure input and weight1 are compatible for matrix multiplication
    assert input.shape[-1] == weight1.shape[0], "Incompatible dimensions for matrix multiplication"

    # Ensure weight2 and bias are broadcastable to the output shape
    output_shape = (*input.shape[:-1], weight1.shape[1])
    weight2 = weight2.expand(output_shape)
    bias = bias.expand(output_shape)

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    # Launch the Triton kernel
    grid = lambda META: (triton.cdiv(input.numel() // input.shape[-1], META['BLOCK_SIZE']),)
    combined_activation_kernel[grid](
        input,  # Input tensor
        weight1,  # Weight matrix 1
        weight2,  # Weight matrix 2
        bias,  # Bias tensor
        out,  # Output tensor
        input.shape[-2],  # Number of elements in the last dimension of X
        input.shape[-1],  # Number of elements in the second-to-last dimension of X
        weight1.shape[1],  # Number of elements in the last dimension of Y
        input.stride(-1),  # Stride for X in the last dimension
        input.stride(-2),  # Stride for X in the second-to-last dimension
        weight1.stride(-1),  # Stride for W1 in the last dimension
        weight2.stride(-1),  # Stride for W2 in the last dimension
        out.stride(-1),  # Stride for Y in the last dimension
        BLOCK_SIZE=128,  # Block size
    )

    return out
