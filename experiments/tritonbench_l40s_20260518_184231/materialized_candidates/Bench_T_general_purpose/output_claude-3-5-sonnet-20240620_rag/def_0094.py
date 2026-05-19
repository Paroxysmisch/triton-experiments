import torch
import triton
import triton.language as tl

@triton.jit
def apply_linear(input, weight, bias, size):
    """
    Applies a linear transformation.

    Args:
        input: Input tensor.
        weight: Weight tensor.
        bias: Bias tensor.
        size: Size of the input.

    Returns:
        Transformed tensor.
    """
    # Linear transformation
    output = tl.dot(input, weight.T)
    if bias is not None:
        output += bias
    return output

@triton.jit
def apply_sigmoid(input):
    """
    Applies the sigmoid activation function.

    Args:
        input: Input tensor.

    Returns:
        Sigmoid activated tensor.
    """
    return 1 / (1 + tl.exp(-input))

@triton.jit
def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
    """
    Applies a linear transformation followed by a sigmoid activation and dropout.

    Args:
        input: Input tensor of shape (*, in_features).
        weight: Weight tensor of shape (out_features, in_features).
        bias: Bias tensor of shape (out_features). Default is None.
        p: Probability of an element to be zeroed in dropout. Default: 0.5.
        training: If True, applies dropout during training. Default: True.
        inplace: If True, performs the operation in-place. Default: False.

    Returns:
        Output tensor after applying linear transformation, sigmoid, and dropout.
    """
    # Apply linear transformation
    linear_output = apply_linear(input, weight, bias, input.shape[-1])
    
    # Apply sigmoid activation
    activated_output = apply_sigmoid(linear_output)

    # Apply dropout if in training mode
    if training:
        drop_p = tl.tensor(p, dtype=tl.float32)
        seed = tl.random_seed()
        offset = tl.arange(0, activated_output.shape[0])
        activated_output = apply_dropout(activated_output, drop_p, seed, offset)

    return activated_output
