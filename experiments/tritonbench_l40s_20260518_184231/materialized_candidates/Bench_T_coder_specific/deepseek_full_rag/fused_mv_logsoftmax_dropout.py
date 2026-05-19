Args:
        prev_ema: Previous exponential moving average.
        new_val: New value to update the exponential moving average with.
        momentum: Momentum to use in the update.

    Returns:
        Updated exponential moving average.
    """
    return (1 - momentum) * prev_ema + momentum * new_val

@triton.jit
def standardize(input, mean, inv_std, inplace: tl.constexpr):
    """
    Standardizes the input.

    Args:
        input: Input to standardize.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        mean: Mean to subtract from the input.
            The mean must be of shape [BLOCK_SIZE1].
        inv_std: Inverse standard deviation to multiply the input by.
            The inverse standard deviation must be of shape [BLOCK_SIZE1].
        inplace: Flag for performing the operation inplace.

    Returns:
        Standardized input.
    """
    input = input.to(tl.float32)

    if inplace:
        output = input
    else:
        output = tl.zeros_like(input)

    output = output - mean[:, None]
    output = output * inv_std[:, None]

    return output

@triton.jit
def calc_l1l2_norm(input, order: tl.constexpr, mask: tl.constexpr):
    """
    Calculates the L1 or L2 norm of the input.

    Args:
        input: Input whose L1 or L2 norm is calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        order: Order of the norm to calculate.
            Options are 1 for L1 norm and 2 for L2 norm.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        L1 or L2 norm of the input.
    """
    input = input.to(tl.float32)

    if order == 1:
        norm = tl.sum(tl.abs(input), axis=1)

    elif order == 2:
        norm = tl.sqrt(tl.sum(input * input, axis=1))

    else:
        raise ValueError("Invalid order")

    return tl.where(mask, norm, 0.)

@triton.jit
def negative_log_likelihood(logits, labels, ignore_index: tl.constexpr):
    """
    Calculates the mean negative log likelihood loss.

    Args:
        logits: Logits to use for calculating the loss.
            The logits must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        labels: Labels corresponding to the logits.
            The labels must be of shape [BLOCK_SIZE1].
        ignore_index: Index to ignore in the labels.

    Returns:
        Mean negative log likelihood loss.
    """
    logits = logits.to(tl.float32)
    labels = labels.to(tl.int32)

    loss = tl.where(
        labels != ignore_index,
        tl.sum(tl.log(tl.sum(tl.exp(logits), axis=1)) - logits[range(tl.num_programs(0)), labels]),
        0.,
    )

    return loss / tl.num_programs(0)

@triton.jit
def cross_entropy(logits, labels, weight, ignore_index: tl.constexpr):
    """
    Calculates the mean cross entropy loss.

    Args:
        logits: Logits to use for calculating the loss.
            The logits must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        labels: Labels corresponding to the logits.
            The labels must be of shape [BLOCK_SIZE1].
        weight: Weight to apply to the loss.
        ignore_index: Index to ignore in the labels.

    Returns:
        Mean cross entropy loss.
    """
    logits = logits.to(tl.float32)
    labels = labels.to(tl.int32)
    weight = weight.to(tl.float32)

    loss = tl.where(
        labels != ignore_index,
        tl.sum(weight * (tl.log(tl.sum(tl.exp(logits), axis=1)) - logits[range(tl.num_programs(0)), labels])),
        0.,
    )

    return loss / tl.num_programs(0)
