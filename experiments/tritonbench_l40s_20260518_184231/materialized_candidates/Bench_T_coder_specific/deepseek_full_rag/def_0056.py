std: Inverse standard deviation of input.
        weight: Weights for standardized input.
        bias: Bias vector added to standardized input.

    Returns:
        Standardized input.
    """
    return (input - mean) * inv_std * weight + bias

@triton.jit
def calc_l1l2_norm(input, p, dim, keepdim: tl.constexpr):
    """
    Calculates the L1 or L2 norm of the input along a specific dimension.

    Args:
        input: Input whose L1 or L2 norm is calculated.
        p: Order of the norm.
        dim: Dimension along which the norm is calculated.
        keepdim: Flag for keeping the dimension in the output.

    Returns:
        L1 or L2 norm of the input along the specified dimension.
    """
    return tl.sum(tl.pow(tl.abs(input), p), dim, keepdim=keepdim)

@triton.jit
def calc_nll_loss(input, target, ignore_index, reduction: tl.constexpr):
    """
    Calculates the negative log likelihood loss for the input given the target.

    Args:
        input: Input to the loss function.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target of the input.
            The target must be of shape [BLOCK_SIZE1].
        ignore_index: Target value to ignore.
        reduction: Reduction strategy for the output.
            Options are 'none', 'mean', and 'sum'.

    Returns:
        Negative log likelihood loss for the input given the target.
    """
    input = input.to(tl.float32)

    target_mask = target != ignore_index
    target = tl.where(target_mask, target, 0)

    loss = -tl.sum(tl.where(target_mask, input[0, target], 0.)) / target_mask.sum()

    if reduction == "mean":
        loss = loss / target_mask.shape[0]
    elif reduction == "sum":
        loss = loss * target_mask.shape[0]

    return loss

@triton.jit
def calc_cross_entropy_loss(input, target, ignore_index, weight, reduction: tl.constexpr):
    """
    Calculates the cross entropy loss for the input given the target.

    Args:
        input: Input to the loss function.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target of the input.
            The target must be of shape [BLOCK_SIZE1].
        ignore_index: Target value to ignore.
        weight: Weights for the loss function.
            The weight must be of shape [BLOCK_SIZE2].
        reduction: Reduction strategy for the output.
            Options are 'none', 'mean', and 'sum'.

    Returns:
        Cross entropy loss for the input given the target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.int32)

    target_mask = target != ignore_index
    target = tl.where(target_mask, target, 0)

    loss = -tl.sum(tl.where(target_mask, input[0, target] * weight[target], 0.)) / target_mask.sum()

    if reduction == "mean":
        loss = loss / target_mask.shape[0]
    elif reduction == "sum":
        loss = loss * target_mask.shape[0]

    return loss
