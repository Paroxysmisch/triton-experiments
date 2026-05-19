import triton
import triton.language as tl

@triton.jit
def fused_hardsigmoid_batch_norm(
    x_ptr: tl.tensor,
    running_mean_ptr: tl.tensor,
    running_var_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    out_ptr: tl.tensor,
    x_shape: tl.constexpr,
    C: tl.constexpr,
    training: tl.constexpr,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    inplace: tl.constexpr,
):
    """
    Applies Batch Normalization followed by the Hardsigmoid activation function on the input tensor `x`.

    Args:
        x_ptr: Pointer to the input tensor for batch normalization and activation.
        running_mean_ptr: Pointer to the running mean buffer (persistent).
        running_var_ptr: Pointer to the running variance buffer (persistent).
        weight_ptr: Pointer to the learnable weight of size C for the normalized tensor.
        bias_ptr: Pointer to the learnable bias of size C for the normalized tensor.
        out_ptr: Pointer to the output tensor where the result will be stored.
        x_shape: Shape of the input tensor.
        C: Number of channels in the input tensor.
        training: Flag for training mode, used to update running estimates.
        momentum: The value for the running mean and variance momentum.
        eps: Small constant added to variance to improve numerical stability.
        inplace: If True, perform Hardsigmoid in-place.
    """

    # Extract dimensions
    N, _, _ = x_shape

    # Load input tensor
    x = tl.load(x_ptr + tl.program_id(0))

    # Calculate mean and variance
    mean = tl.zeros(C, dtype=tl.float32)
    var = tl.zeros(C, dtype=tl.float32)
    for i in range(N):
        mean += x[i]
        var += x[i] * x[i]
    mean /= N
    var /= N - 1

    # Update running estimates if in training mode
    if training:
        running_mean = tl.load(running_mean_ptr)
        running_var = tl.load(running_var_ptr)
        running_mean = (1 - momentum) * running_mean + momentum * mean
        running_var = (1 - momentum) * running_var + momentum * var
        tl.store(running_mean_ptr, running_mean)
        tl.store(running_var_ptr, running_var)

    # Normalize the input
    normalized_x = (x - mean) / tl.sqrt(var + eps)

    # Apply learnable weight and bias if provided
    if weight_ptr is not None and bias_ptr is not None:
        weight = tl.load(weight_ptr)
        bias = tl.load(bias_ptr)
        normalized_x *= weight
        normalized_x += bias

    # Apply Hardsigmoid activation
    hardsigmoid_slope = 0.2
    hardsigmoid_shift = 0.5
    if inplace:
        out = normalized_x
        out[out < 0] = 0
        out[out > 1] = 1
        out *= hardsigmoid_slope
        out += hardsigmoid_shift
        tl.store(out_ptr + tl.program_id(0), out)
    else:
        out = tl.clip(normalized_x * hardsigmoid_slope + hardsigmoid_shift, 0, 1)
        tl.store(out_ptr + tl.program_id(0), out)


# Example usage:
# Assuming you have the necessary tensors and shapes
# x = ...
# running_mean = ...
# running_var = ...
# weight = ...
# bias = ...
# out = ...
# fused_hardsigmoid_batch_norm[x, running_mean, running_var, weight, bias, out, x.shape, C, training, momentum, eps, inplace]
