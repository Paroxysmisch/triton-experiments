.jit
def update_ema(input, prev_val, momentum):
    """
    Updates value using exponential moving average update.

    Args:
        input: Input used to update value.
            The input must be of the same shape as the mask.
        prev_val: Previous value to update.
        momentum: Momentum coefficient.

    Returns:
        Updated value.
    """
    return (1 - momentum) * prev_val + momentum * input

@triton.jit
def standardize(input, mean, inv_std, eps, log: tl.constexpr):
    """
    Standardizes the input using mean and inverse standard deviation.

    Args:
        input: Input to standardize.
            The input must be of the same shape as the mean and inv_std.
        mean: Mean to subtract.
            The mean must be of the same shape as the input.
        inv_std: Inverse standard deviation to multiply.
            The inv_std must be of the same shape as the input.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        log: Flag for indicating if the log should be taken.

    Returns:
        Standardized input.
    """
    input = input.to(tl.float32)

    if log:
        output = (input - mean) * inv_std
    else:
        output = (input - mean) * inv_std

    return output

@triton.jit
def l1_l2_norm_loss(input, target, reduction: tl.constexpr, p: tl.constexpr):
    """
    Calculates the L1/L2 norm loss between the input and target.

    Args:
        input: Input tensor.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target tensor.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        reduction: Reduction strategy for the output.
            Options are 'none', 'mean', and 'sum'.
        p: Order of the norm.

    Returns:
        L1/L2 norm loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    diff = input - target
    output = tl.sum(tl.pow(tl.abs(diff), p))

    if reduction == "mean":
        output /= tl.numel(input)
    elif reduction == "sum":
        output = tl.sum(output)

    return output

@triton.jit
def neg_log_likelihood_loss(input, target, eps: tl.constexpr):
    """
    Calculates the negative log likelihood loss between the input and target.

    Args:
        input: Input tensor.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target tensor.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        eps: Epsilon added in the log to avoid log(0) = -inf.

    Returns:
        Negative log likelihood loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    input_clamped = tl.maximum(input, eps)
    log_input_clamped = tl.log(input_clamped)

    output = -tl.sum(target * log_input_clamped)

    return output

@triton.jit
def cross_entropy_loss(input, target, last_dim_mask: tl.constexpr, eps: tl.constexpr):
    """
    Calculates the cross entropy loss between the input and target.

    Args:
        input: Input tensor.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target tensor.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the calculations.
            The mask must be of shape [BLOCK_SIZE2].
        eps: Epsilon added in the log to avoid log(0) = -inf.

    Returns:
        Cross entropy loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    input_clamped = tl.maximum(input, eps)
    log_input_clamped = tl.log(input_clamped)

    output = -tl.sum(target * log_input_clamped, axis=1)

    return output

@triton.jit
def sigmoid_batch_norm(input, running_mean, running_var, weight, bias, training: tl.constexpr, momentum, eps):
    """
    Applies batch normalization to the input tensor,
    followed by applying the sigmoid activation function element-wise.

    Args:
        input: Input tensor of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        running_mean: Running mean of the input channels.
            The running mean must be of shape [BLOCK_SIZE2].
        running_var: Running variance of the input channels.
            The running variance must be of shape [BLOCK_SIZE2].
        weight: Weight tensor for scaling.
            The weight tensor must be of shape [BLOCK_SIZE2].
        bias: Bias tensor for shifting.
            The bias tensor must be of shape [BLOCK_SIZE2].
        training: Flag for updating the running mean and variance.
        momentum: Momentum coefficient for the running mean and variance.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.

    Returns:
        Input tensor normalized by batch normalization
        and then passed through a sigmoid activation function.
    """
    input = input.to(tl.float32)
    running_mean = running_mean.to(tl.float32)
    running_var = running_var.to(tl.float32)
    weight = weight.to(tl.float32)
    bias = bias.to(tl.float32)

    if training:
        mean = tl.sum(input, axis=0) / input.shape[0]
        diff = input - mean
        var = tl.sum(diff * diff, axis=0) / input.shape[0]

        running_mean = update_ema(mean, running_mean, momentum)
        running_var = update_ema(var, running_var, momentum)

    else:
        mean = running_mean
        var = running_var

    inv_std = tl.rsqrt(var + eps)
    output = (input - mean) * inv_std
    output = output * weight + bias
    output = tl.sigmoid(output)

    return output
