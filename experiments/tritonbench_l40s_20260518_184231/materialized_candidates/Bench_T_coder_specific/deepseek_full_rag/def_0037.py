and inverse standard deviation.

    Args:
        input: Input to standardize.
            The input must be of the same shape as the mean and inverse standard deviation.
        mean: Mean of the input.
            The mean must be of the same shape as the input.
        inv_std: Inverse standard deviation of the input.
            The inverse standard deviation must be of the same shape as the input.
        weight: Weight used to rescale the standardized input.
            The weight must be of the same shape as the input.
        bias: Bias used to shift the standardized input.
            The bias must be of the same shape as the input.

    Returns:
        Standardized input.
    """
    return weight * (input - mean) * inv_std + bias

@triton.jit
def calc_l1l2_norm_loss(input, target, weight, reduction: tl.constexpr):
    """
    Calculates the mean L1/L2 norm loss between the input and target.

    Args:
        input: Input to error.
            The input must be of the same shape as the target.
        target: Target to compare the input to.
            The target must be of the same shape as the input.
        weight: Weight used to scale the loss.
            The weight must be of the same shape as the input.
        reduction: Reduction method to apply to the loss.
            Options are 'none' for no reduction, 'mean' for averaging the loss,
            and 'sum' for summing the loss.

    Returns:
        L1/L2 norm loss between the input and target.
    """
    diff = input - target
    abs_diff = tl.abs(diff)
    l1_loss = tl.sum(weight * abs_diff, axis=0)

    if reduction == 'mean':
        l1_loss = l1_loss / tl.sum(weight)

    l2_loss = tl.sqrt(tl.sum(weight * diff * diff, axis=0))

    if reduction == 'mean':
        l2_loss = l2_loss / tl.sum(weight)

    return l1_loss, l2_loss

@triton.jit
def calc_neg_log_likelihood_loss(input, target, mask, reduction: tl.constexpr):
    """
    Calculates the mean negative log likelihood loss between the input and target.

    Args:
        input: Input to error.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target to compare the input to.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        mask: Mask indicating which elements should be included in the loss.
            The mask must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        reduction: Reduction method to apply to the loss.
            Options are 'none' for no reduction, 'mean' for averaging the loss,
            and 'sum' for summing the loss.

    Returns:
        Negative log likelihood loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)
    mask = mask.to(tl.float32)

    loss = -tl.sum(mask * input * target, axis=0) / tl.sum(mask, axis=0)

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        return tl.sum(loss)

    return loss

@triton.jit
def calc_cross_entropy_loss(input, target, weight, ignore_index, reduction: tl.constexpr):
    """
    Calculates the mean cross entropy loss between the input and target.

    Args:
        input: Input to error.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target to compare the input to.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        weight: Weight used to scale the loss.
            The weight must be of shape [BLOCK_SIZE2].
        ignore_index: Index to ignore in the loss calculation.
        reduction: Reduction method to apply to the loss.
            Options are 'none' for no reduction, 'mean' for averaging the loss,
            and 'sum' for summing the loss.

    Returns:
        Cross entropy loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)
    weight = weight.to(tl.float32)

    loss = tl.sum(weight * (target * tl.log(input) + (1 - target) * tl.log(1 - input)), axis=0)

    if ignore_index != -1:
        loss = loss * (1 - tl.sum(target[:, ignore_index]))

    if reduction == 'none':
        return loss

    elif reduction == 'sum':
        return tl.sum(loss)

    return loss / tl.sum(target)
