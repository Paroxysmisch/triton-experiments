import triton
import triton.language as tl

@triton.jit
def linear_sigmoid_dropout_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr, dropout_mask_ptr,
    in_features, out_features, batch_size, drop_p, seed, training, inplace,
    BLOCK_SIZE: tl.constexpr
):
    """
    Applies a linear transformation followed by a sigmoid activation and dropout.

    Args:
        input_ptr: Pointer to the input tensor of shape [batch_size, in_features].
        weight_ptr: Pointer to the weight tensor of shape [out_features, in_features].
        bias_ptr: Pointer to the bias tensor of shape [out_features]. Can be None.
        output_ptr: Pointer to the output tensor of shape [batch_size, out_features].
        dropout_mask_ptr: Pointer to the dropout mask tensor of shape [batch_size, out_features].
        in_features: Number of input features.
        out_features: Number of output features.
        batch_size: Number of elements in the batch.
        drop_p: Probability of an element to be zeroed in dropout.
        seed: Seed for generating the dropout mask.
        training: If True, applies dropout during training.
        inplace: If True, performs the operation in-place.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    batch_idx = pid // (out_features // BLOCK_SIZE)
    out_feature_idx = (pid % (out_features // BLOCK_SIZE)) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_feature_idx = tl.arange(0, in_features)

    # Load input and weight
    input = tl.load(input_ptr + batch_idx * in_features + in_feature_idx, mask=in_feature_idx < in_features)
    weight = tl.load(weight_ptr + out_feature_idx[:, None] * in_features + in_feature_idx[None, :], mask=out_feature_idx[:, None] < out_features, other=0.0)

    # Compute linear transformation
    linear_output = tl.sum(input[None, :] * weight, axis=1)

    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + out_feature_idx, mask=out_feature_idx < out_features)
        linear_output += bias

    # Apply sigmoid activation
    sigmoid_output = 1 / (1 + tl.exp(-linear_output))

    # Apply dropout if training is True
    if training:
        random = tl.rand(seed, batch_idx * out_features + out_feature_idx)
        dropout_mask = random > drop_p
        sigmoid_output = tl.where(dropout_mask, sigmoid_output / (1 - drop_p), 0.0)

    # Store the output
    tl.store(output_ptr + batch_idx * out_features + out_feature_idx, sigmoid_output, mask=out_feature_idx < out_features)

    # Store the dropout mask if needed
    if training and dropout_mask_ptr is not None:
        tl.store(dropout_mask_ptr + batch_idx * out_features + out_feature_idx, dropout_mask, mask=out_feature_idx < out_features)

import torch
import triton
import triton.language as tl

def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
    """
    Applies a linear transformation followed by a sigmoid activation and dropout.

    Args:
        input: Input tensor of shape (*, in_features).
        weight: Weight tensor of shape (out_features, in_features).
        bias: Bias tensor of shape (out_features). Default is None.
        p: Probability of an element to be zeroed in dropout. Default: 0.5
        training: If True, applies dropout during training. Default: True
        inplace: If True, performs the operation in-place. Default: False

    Returns:
        Output tensor of shape (*, out_features).
    """
    in_features = input.shape[-1]
    out_features = weight.shape[0]
    batch_size = input.numel() // in_features

    # Ensure input is contiguous
    input = input.contiguous()

    # Allocate output tensor
    if inplace:
        output = input.view(-1, in_features)
    else:
        output = torch.empty((batch_size, out_features), device=input.device, dtype=input.dtype)

    # Allocate dropout mask if needed
    dropout_mask = None
    if training:
        dropout_mask = torch.empty((batch_size, out_features), device=input.device, dtype=torch.bool)

    # Launch the kernel
    grid = (batch_size * (out_features // 32),)
    linear_sigmoid_dropout_kernel[grid](
        input, weight, bias, output, dropout_mask,
        in_features, out_features, batch_size, p, 42, training, inplace, BLOCK_SIZE=32
    )

    return output.view(*input.shape[:-1], out_features)

# Example usage
input = torch.randn(10, 5, device='cuda')
weight = torch.randn(3, 5, device='cuda')
bias = torch.randn(3, device='cuda')
output = dropout_sigmoid_linear(input, weight, bias, p=0.5, training=True, inplace=False)
print(output)
