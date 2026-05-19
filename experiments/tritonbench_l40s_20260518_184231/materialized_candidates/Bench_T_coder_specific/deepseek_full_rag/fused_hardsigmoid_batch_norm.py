: Previous exponential moving average.
        new_val: New value to update the exponential moving average with.
        momentum: Momentum for exponential moving average update.

    Returns:
        Updated exponential moving average.
    """
    return (1 - momentum) * prev_ema + momentum * new_val

@triton.jit
def standardize(input, mean, inv_std, last_dim_mask: tl.constexpr):
    """
    Standardizes the input using given mean and inverse standard deviation.

    Args:
        input: Input to standardize.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        mean: Mean to standardize the input with.
            The mean must be of shape [BLOCK_SIZE1].
        inv_std: Inverse standard deviation to standardize the input with.
            The inverse standard deviation must be of shape [BLOCK_SIZE1].
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the calculations.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Standardized input.
    """
    input = input.to(tl.float32)

    output = (input - mean[:, None]) * inv_std[:, None]
    return tl.where(last_dim_mask[None, :], output, 0)

@triton.jit
def l1_l2_norm_loss(input1, input2, p, last_dim_mask: tl.constexpr):
    """
    Calculates the L1/L2 norm loss between two inputs.

    Args:
        input1: First input.
            The first input must be of the same shape as the second input.
        input2: Second input.
            The second input must be of the same shape as the first input.
        p: Order of the norm.
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the calculations.
            The mask must be of the same shape as the input.

    Returns:
        L1/L2 norm loss between the two inputs.
    """
    input1 = input1.to(tl.float32)
    input2 = input2.to(tl.float32)

    diff = tl.where(last_dim_mask, input1 - input2, 0)
    output = tl.sum(tl.pow(tl.abs(diff), p)) / tl.sum(tl.cast(last_dim_mask, tl.float32))

    return output

@triton.jit
def neg_log_likelihood_loss(input, target, last_dim_mask: tl.constexpr):
    """
    Calculates the negative log likelihood loss between an input and a target.

    Args:
        input: Input.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the calculations.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Negative log likelihood loss between the input and the target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    input_clamped = tl.where(last_dim_mask, tl.minimum(tl.maximum(input, 0), 1 - 1e-5), 0)
    output = -tl.sum(target * tl.log(input_clamped)) / tl.sum(tl.cast(last_dim_mask, tl.float32))

    return output

@triton.jit
def cross_entropy_loss(input, target, last_dim_mask: tl.constexpr):
    """
    Calculates the cross entropy loss between an input and a target.

    Args:
        input: Input.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the calculations.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Cross entropy loss between the input and the target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    input_clamped = tl.where(last_dim_mask, tl.minimum(tl.maximum(input, 0), 1 - 1e-5), 0)
    output = -tl.sum(target * tl.log(input_clamped)) / tl.sum(tl.cast(last_dim_mask, tl.float32))

    return output
