dimension of the input.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        last_dim_mask: Mask to apply to the mean and variance
            calculation only over the last dimension.

    Returns:
        Mean and inverse standard deviation of the input.
    """
    if last_dim_mask:
        input = input.to(tl.float32)

        input = tl.sum(input, axis=0)
        count = tl.cdiv(last_dim, 2) * 2

    else:
        input = input.to(tl.float32)

        input = tl.sum(input, axis=-1)[:, None]
        count = last_dim

    mean = input / count
    mean2 = tl.sum(mean * mean, axis=0)
    inv_std = tl.math.rsqrt(mean2 + eps)

    return mean, inv_std

@triton.jit
def welford_update(input, mean, inv_std, eps, last_dim, last_dim_mask: tl.constexpr):
    """
    Updates the Welford statistics given an input.

    Args:
        input: Input whose Welford statistics are calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        mean: Mean of the input.
        inv_std: Inverse standard deviation of the input.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        last_dim: Size of the last dimension of the input.
        last_dim_mask: Mask to apply to the mean and variance
            calculation only over the last dimension.

    Returns:
        Updated mean and inverse standard deviation.
    """
    if last_dim_mask:
        input = input.to(tl.float32)

        input = tl.sum(input, axis=0)
        count = tl.cdiv(last_dim, 2) * 2

    else:
        input = input.to(tl.float32)

        input = tl.sum(input, axis=-1)[:, None]
        count = last_dim

    new_mean = input / count
    delta = new_mean - mean
    mean = mean + delta * inv_std
    inv_std = inv_std * tl.math.rsqrt(mean2 + eps)

    return mean, inv_std

@triton.jit
def exp_avg_update(input, old, inv_std, eps, last_dim, last_dim_mask: tl.constexpr):
    """
    Updates the exponential average given an input.

    Args:
        input: Input whose exponential average is calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        old: Old exponential average.
        inv_std: Inverse standard deviation of the input.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        last_dim: Size of the last dimension of the input.
        last_dim_mask: Mask to apply to the mean and variance
            calculation only over the last dimension.

    Returns:
        Updated exponential average.
    """
    if last_dim_mask:
        input = input.to(tl.float32)

        input = tl.sum(input, axis=0)
        count = tl.cdiv(last_dim, 2) * 2

    else:
        input = input.to(tl.float32)

        input = tl.sum(input, axis=-1)[:, None]
        count = last_dim

    new_avg = input / count
    delta = new_avg - old
    output = old + delta * inv_std

    return output

@triton.jit
def standardize(input, mean, inv_std):
    """
    Standardizes the input given a mean and inverse standard deviation.

    Args:
        input: Input to standardize.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        mean: Mean of the input.
        inv_std: Inverse standard deviation of the input.

    Returns:
        Standardized input.
    """
    input = input.to(tl.float32)

    output = (input - mean) * inv_std

    return output

@triton.jit
def calc_l1l2_norm(input, reduction_dim, p: tl.constexpr, keepdim: tl.constexpr):
    """
    Calculates the L1 or L2 norm of the input along a specified dimension.

    Args:
        input: Input whose L1 or L2 norm is calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        reduction_dim: Dimension along which the norm is calculated.
        p: Order of the norm.
            Options are 1 for L1 norm and 2 for L2 norm.
        keepdim: Flag for keeping the dimension of the input in the output.

    Returns:
        L1 or L2 norm of the input along the specified dimension.
    """
    if p == 1:
        output = tl.sum(tl.abs(input), axis=reduction_dim, keepdim=keepdim)

    elif p == 2:
        output = tl.sum(input * input, axis=reduction_dim, keepdim=keepdim)
        output = tl.sqrt(output)

    return output

@triton.jit
def negative_log_likelihood(logits, targets, log_2pi, log_var_clamp, fp16: tl.constexpr):
    """
    Calculates the mean negative log likelihood of the logits given the targets.

    Args:
        logits: Logits to use for prediction.
            The logits must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        targets: Targets to use for calculating the negative log likelihood.
            The targets must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        log_2pi: Logarithm of 2Pi.
        log_var_clamp: Clamping value for the variance in the calculation
            of the negative log likelihood.
        fp16: Flag for converting operands to FP16.

    Returns:
        Mean negative log likelihood of the logits given the targets.
    """
    if fp16:
        logits = logits.to(tl.float16)
        targets = targets.to(tl.float16)

    logits = logits - 0.5 * log_2pi
    logits = logits - 0.5 * tl.math.log1p(tl.exp(-logits))

    targets = targets.to(tl.float32)

    logits = logits.to(tl.float32)
    log_var = tl.log1p(tl.exp(-logits))
    log_var = tl.where(log_var <= log_var_clamp, log_var_clamp, log_var)

    loss = 0.5 * tl.sum((targets - logits) ** 2 * log_var.to(tl.float32), axis=1)
    loss += 0.5 * tl.sum(log_var, axis=1)

    return loss

@triton.jit
def cross_entropy(logits, targets, ignore_index, fp16: tl.constexpr):
    """
    Calculates the mean cross entropy loss of the logits given the targets.

    Args:
        logits: Logits to use for prediction.
            The logits must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        targets: Targets to use for calculating the cross entropy loss.
            The targets must be of shape [BLOCK_SIZE1].
        ignore_index: Target value to ignore in the calculation of the cross entropy loss.
        fp16: Flag for converting operands to FP16.

    Returns:
        Mean cross entropy loss of the logits given the targets.
    """
    if fp16:
        logits = logits.to(tl.float16)
