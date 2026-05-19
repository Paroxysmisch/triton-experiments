inv_std: Inverse standard deviation of input.
        weight: Weights used to scale the standardized input.
        bias: Bias vector added to the scaled standardized input.

    Returns:
        Standardized input scaled by weights and biased.
    """
    return (input - mean) * inv_std * weight + bias

@triton.jit
def calc_l1l2_norm(input, p: tl.constexpr, dim, keepdim: tl.constexpr):
    """
    Calculates the L1 or L2 norm of the input.

    Args:
        input: Input whose L1 or L2 norm is calculated.
        p: Order of the norm.
        dim: Dimension along which to calculate the norm.
        keepdim: Flag for keeping the dimension in the output.

    Returns:
        L1 or L2 norm of the input.
    """
    if p == 1:
        return tl.sum(tl.abs(input), dim=dim, keepdim=keepdim)

    elif p == 2:
        return tl.sum(input * input, dim=dim, keepdim=keepdim) ** 0.5

    else:
        raise ValueError("p must be 1 or 2")

@triton.jit
def neg_log_likelihood(input, target, ignore_index: tl.constexpr, reduction: tl.constexpr):
    """
    Calculates the negative log likelihood loss of the input.

    Args:
        input: Input whose negative log likelihood loss is calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target of the input.
            The target must be of shape [BLOCK_SIZE1]
            and have the same dtype as the input.
        ignore_index: Index to ignore in the calculation of the loss.
        reduction: Reduction method for the output.
            Options are 'none', 'mean', 'sum', and 'batchmean'.

    Returns:
        Negative log likelihood loss of the input.
    """
    log_input = tl.where(input != 0, tl.log(input), 0)
    nll = -tl.sum(input * target, axis=1)

    if ignore_index != -1000:
        target = target == ignore_index
        nll += tl.sum(input * target, axis=1)

    if reduction == "mean":
        nll = nll / tl.sum(1 - target, axis=1)

    elif reduction == "sum":
        nll = tl.sum(nll)

    elif reduction == "batchmean":
        nll = tl.sum(nll) / target.shape[0]

    return nll

@triton.jit
def cross_entropy(input, target, ignore_index: tl.constexpr, reduction: tl.constexpr):
    """
    Calculates the cross entropy loss of the input.

    Args:
        input: Input whose cross entropy loss is calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target of the input.
            The target must be of shape [BLOCK_SIZE1]
            and have the same dtype as the input.
        ignore_index: Index to ignore in the calculation of the loss.
        reduction: Reduction method for the output.
            Options are 'none', 'mean', 'sum', and 'batchmean'.

    Returns:
        Cross entropy loss of the input.
    """
    input = input.to(tl.float32)

    log_input = tl.where(input != 0, tl.log(input), 0)
    ce = tl.sum(target * log_input, axis=1)

    if ignore_index != -1000:
        target = target == ignore_index
        ce += tl.sum(input * target, axis=1)

    if reduction == "mean":
        ce = ce / tl.sum(1 - target, axis=1)

    elif reduction == "sum":
        ce = tl.sum(ce)

    elif reduction == "batchmean":
        ce = tl.sum(ce) / target.shape[0]

    return ce
