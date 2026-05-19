val, decay):
    """
    Updates exponential moving average (EMA) with given decay.

    Args:
        prev_ema: Previous EMA statistic to update.
        new_val: New value to incorporate into the EMA.
            The new value must be of the same shape as the EMA.
        decay: Decay factor to use in updating the EMA.

    Returns:
        Updated exponential moving average.
    """
    return decay * prev_ema + (1 - decay) * new_val

@triton.jit
def standardize(input, mean, inv_std, mask: tl.constexpr):
    """
    Standardizes the input using given mean and inverse standard deviation.

    Args:
        input: Input to standardize.
            The input must be of the same shape as the mask.
        mean: Mean to subtract from the input.
            The mean must be of the same shape as the input.
        inv_std: Inverse standard deviation to multiply the input by.
            The inverse standard deviation must be a scalar or of the
            same shape as the input, in which case it must be a
            broadcastable shape.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of the same shape as the input.

    Returns:
        Standardized input.
    """
    input = input.to(tl.float32)

    return tl.where(mask, (input - mean) * inv_std, 0.)

@triton.jit
def calc_loss(input, target, reduction: tl.constexpr):
    """
    Calculates the loss between the input and target.

    Args:
        input: Input tensor from the model.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target tensor for the model to be compared against.
            The target must be of the same shape as the input.
        reduction: Reduction strategy for the loss.
            Options are 'none', 'mean', and 'sum'.

    Returns:
        Loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    loss = tl.sum((input - target) * (input - target))

    if reduction == "mean":
        loss = loss / tl.num_programs(0)

    elif reduction == "sum":
        loss = loss

    return loss

@triton.jit
def calc_l1_loss(input, target, reduction: tl.constexpr):
    """
    Calculates the L1 loss between the input and target.

    Args:
        input: Input tensor from the model.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target tensor for the model to be compared against.
            The target must be of the same shape as the input.
        reduction: Reduction strategy for the loss.
            Options are 'none', 'mean', and 'sum'.

    Returns:
        L1 loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    loss = tl.sum(tl.abs(input - target))

    if reduction == "mean":
        loss = loss / tl.num_programs(0)

    elif reduction == "sum":
        loss = loss

    return loss

@triton.jit
def calc_l2_loss(input, target, reduction: tl.constexpr):
    """
    Calculates the L2 loss between the input and target.

    Args:
        input: Input tensor from the model.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target tensor for the model to be compared against.
            The target must be of the same shape as the input.
        reduction: Reduction strategy for the loss.
            Options are 'none', 'mean', and 'sum'.

    Returns:
        L2 loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    loss = tl.sum((input - target) * (input - target))

    if reduction == "mean":
        loss = loss / tl.num_programs(0)

    elif reduction == "sum":
        loss = loss

    return loss

@triton.jit
def calc_neg_ll_loss(input, target, ignore_index: tl.constexpr, reduction: tl.constexpr):
    """
    Calculates the negative log likelihood loss between the input and target.

    Args:
        input: Input tensor from the model.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target tensor for the model to be compared against.
            The target must be of the same shape as the input.
        ignore_index: Index to ignore in the target.
            The index is not used if it is negative.
        reduction: Reduction strategy for the loss.
            Options are 'none', 'mean', and 'sum'.

    Returns:
        Negative log likelihood loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.int32)

    loss = 0.
    num_pixels = 0

    if ignore_index >= 0:
        mask = target != ignore_index
        input = input.to(tl.float32)
        target = target.to(tl.float32)

        loss += -tl.sum(input[target != ignore_index] * target[target != ignore_index])
        num_pixels += tl.sum(target[target != ignore_index])

    loss += -tl.sum(tl.where(target == ignore_index, 0., input))
    num_pixels += tl.sum(target)

    loss = loss / num_pixels

    if reduction == "mean":
        loss = loss / tl.num_programs(0)

    elif reduction == "sum":
        loss = loss

    return loss

@triton.jit
def calc_cross_entropy_loss(input, target, ignore_index: tl.constexpr, reduction: tl.constexpr):
    """
    Calculates the cross entropy loss between the input and target.

    Args:
        input: Input tensor from the model.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target tensor for the model to be compared against.
            The target must be of the same shape as the input.
        ignore_index: Index to ignore in the target.
            The index is not used if it is negative.
        reduction: Reduction strategy for the loss.
            Options are 'none', 'mean', and 'sum'.

    Returns:
        Cross entropy loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.int32)

    loss = 0.
    num_pixels = 0

    if ignore_index >= 0:
        mask = target != ignore_index
        input = input.to(tl.float32)
        target = target.to(tl.float32)

        loss += -tl.sum(input[mask] * target[mask])
        num_pixels += tl.sum(target[mask])

    loss += -tl.sum(tl.where(target == ignore_index, 0., input))
    num_pixels += tl.sum(target)

    loss = loss / num_pixels

    if reduction == "mean":
        loss = loss / tl.num_programs(0)

    elif reduction == "sum":
        loss = loss

    return loss
