import triton
import triton.language as tl

@triton.jit
def fused_hardsigmoid_batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps, inplace):
    """
    Applies Batch Normalization followed by Hardsigmoid activation.

    Args:
        x: Input tensor for batch normalization and activation.
        running_mean: The running mean buffer (persistent).
        running_var: The running variance buffer (persistent).
        weight: Learnable weight of size C for the normalized tensor.
        bias: Learnable bias of size C for the normalized tensor.
        training: Flag for training mode, used to update running estimates.
        momentum: The value for the running mean and variance momentum.
        eps: Small constant added to variance to improve numerical stability.
        inplace: If True, perform Hardsigmoid in-place.

    Returns:
        Tensor after applying Batch Normalization and Hardsigmoid activation.
    """
    # Normalize the input
    if training:
        mean = tl.mean(x, axis=0)
        var = tl.var(x, axis=0)
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
    else:
        mean = running_mean
        var = running_var

    # Batch normalization
    x_normalized = (x - mean) / tl.sqrt(var + eps)
    
    # Apply weight and bias if provided
    if weight is not None:
        x_normalized = x_normalized * weight
    if bias is not None:
        x_normalized = x_normalized + bias

    # Hardsigmoid activation
    if inplace:
        x_normalized = tl.clip(x_normalized, min=0, max=6) / 6
        x_normalized = tl.clip(x_normalized + 0.5, min=0, max=1)
    else:
        x_normalized = tl.clip(x_normalized, min=0, max=6) / 6
        x_normalized = tl.clip(x_normalized + 0.5, min=0, max=1)

    return x_normalized
