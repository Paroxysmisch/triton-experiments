std * (input - mean) + bias

@triton.jit
def l1l2_loss(input, target, reduction: tl.constexpr, weight: tl.constexpr,
              bias: tl.constexpr, eps: tl.constexpr, p: tl.constexpr):
    """
    Calculates the mean L1 loss or mean L2 loss (MSE) of the input
    with respect to the target.

    Args:
        input: Input whose loss is calculated.
        target: Target input is compared with.
        reduction: Flag for reduction strategy.
            Options are 'none', 'mean', 'sum', and 'batchwise_mean'.
        weight: Weight multiplied by the loss.
        bias: Bias added to the loss.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero for L2 loss.
        p: Order of the norm to use for L norm loss.

    Returns:
        Mean L1 loss or mean L2 loss (MSE) of the input with respect to the target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    diff = input - target
    if p == 1:
        loss = tl.sum(tl.abs(diff))
    elif p == 2:
        loss = tl.sum(diff * diff) / (diff.shape[0] * diff.shape[1])
    else:
        loss = tl.sum(tl.pow(tl.abs(diff) + eps, p)) / (diff.shape[0] * diff.shape[1])

    if reduction == "mean":
        loss = loss / (diff.shape[0] * diff.shape[1])
    elif reduction == "sum":
        pass
    elif reduction == "batchwise_mean":
        loss = loss / diff.shape[0]

    return weight * loss + bias

@triton.jit
def neg_log_likelihood(input, target, ignore_index: tl.constexpr):
    """
    Calculates the mean negative log likelihood loss of the input
    given the target.

    Args:
        input: Input whose loss is calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target input is compared with.
            The target must be of shape [BLOCK_SIZE1].
        ignore_index: Value indicating that a target
            element should be ignored in the calculation.

    Returns:
        Mean negative log likelihood loss of the input given the target.
    """
    log_input = tl.log(input)
    loss = -tl.sum(tl.where(target != ignore_index, log_input[target], 0)) / target.shape[0]

    return loss

@triton.jit
def cross_entropy(input, target, ignore_index: tl.constexpr, label_smoothing: tl.constexpr):
    """
    Calculates the mean cross entropy loss of the input given the target.

    Args:
        input: Input whose loss is calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target input is compared with.
            The target must be of shape [BLOCK_SIZE1].
        ignore_index: Value indicating that a target
            element should be ignored in the calculation.
        label_smoothing: Smoothing parameter used in
            calculating the loss.

    Returns:
        Mean cross entropy loss of the input given the target.
    """
    if label_smoothing > 0:
        input = input * (1 - label_smoothing) + label_smoothing / input.shape[1]
        input = input.to(tl.float32)

    loss = tl.where(target != ignore_index, -tl.sum(input[target]), 0) / target.shape[0]

    return loss
