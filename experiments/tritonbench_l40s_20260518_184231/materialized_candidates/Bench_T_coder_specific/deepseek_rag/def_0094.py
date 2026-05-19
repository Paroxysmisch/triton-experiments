import triton.language as tl
import triton

@triton.jit
def dropout_sigmoid_linear(input, weight, bias, p, training, inplace):
    """
    Applies a linear transformation followed by a sigmoid activation and dropout.

    Args:
        input: Input tensor. The input must be loaded and cannot be a pointer.
        weight: Weight tensor. The weight must be loaded and cannot be a pointer.
        bias: Bias tensor. The bias must be loaded and cannot be a pointer.
        p: Probability of an element to be zeroed in dropout.
        training: If True, applies dropout during training.
        inplace: If True, performs the operation in-place.

    Returns:
        Output tensor.
    """
    # Perform linear transformation
    output = tl.dot(input, weight, bias)

    # Apply dropout
    if training:
        output = apply_dropout(output, p, seed, offset)

    # Apply sigmoid activation
    output = tl.sigmoid(output)

    return output
