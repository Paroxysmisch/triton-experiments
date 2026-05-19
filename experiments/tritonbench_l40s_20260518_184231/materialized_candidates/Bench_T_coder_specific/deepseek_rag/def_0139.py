, inv_std, eps, mask: tl.constexpr):
    """
    Standardizes the input using mean and inverse standard deviation.

    Args:
        input: Input to standardize.
            The input must be of the same shape as the mask.
        mean: Mean to use for standardization.
            The mean must be of the same shape as the input.
        inv_std: Inverse standard deviation to use for standardization.
            The inv_std must be of the same shape as the input.
        eps: Epsilon added in the denominator to avoid division by zero.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of the same shape as the input.

    Returns:
        Standardized input.
    """
    input = input.to(tl.float32)

    return tl.where(mask, (input - mean[:, None]) * inv_std[:, None] + eps, 0.)

@triton.jit
def calc_l1_l2_norm(input1, input2, p, reduction: tl.constexpr):
    """
    Calculates the L1/L2 norm of the input.

    Args:
        input1: First input to calculate the norm of.
            The input must be of the same shape as the second input.
        input2: Second input to calculate the norm of.
            The input must be of the same shape as the first input.
        p: Order of the norm to calculate.
        reduction: Reduction to apply.
            Options are 'none', 'sum', 'mean', 'max', and 'min'.

    Returns:
        L1/L2 norm of the input.
    """
    input1 = input1.to(tl.float32)
    input2 = input2.to(tl.float32)

    norm = tl.where(p == 1, tl.sum(tl.abs(input1 - input2)), 0.)
    norm += tl.where(p == 2, tl.sum((input1 - input2) ** 2), 0.)
    norm = tl.sqrt(norm) if p == 2 else norm

    if reduction == 'sum':
        return tl.sum(norm)
    elif reduction == 'mean':
        return tl.sum(norm) / tl.sum(tl.ones_like(norm))
    elif reduction == 'max':
        return tl.max(norm)
    elif reduction == 'min':
        return tl.min(norm)
    else:
        return norm

@triton.jit
def calc_nll_loss(input1, input2, reduction: tl.constexpr):
    """
    Calculates the negative log likelihood loss of the input.

    Args:
        input1: Predicted log probabilities.
            The input must be of the same shape as the second input.
        input2: True labels.
            The input must be of the same shape as the first input.
        reduction: Reduction to apply.
            Options are 'none', 'sum', 'mean', 'max', and 'min'.

    Returns:
        Negative log likelihood loss of the input.
    """
    input1 = input1.to(tl.float32)
    input2 = input2.to(tl.float32)

    loss = -tl.log(tl.exp(input1)[tl.arange(input1.shape[0]), input2])

    if reduction == 'sum':
        return tl.sum(loss)
    elif reduction == 'mean':
        return tl.sum(loss) / tl.sum(tl.ones_like(loss))
    elif reduction == 'max':
        return tl.max(loss)
    elif reduction == 'min':
        return tl.min(loss)
    else:
        return loss

@triton.jit
def calc_cross_entropy_loss(input1, input2, reduction: tl.constexpr):
    """
    Calculates the cross entropy loss of the input.

    Args:
        input1: Predicted log probabilities.
            The input must be of the same shape as the second input.
        input2: True labels.
            The input must be of the same shape as the first input.
        reduction: Reduction to apply.
            Options are 'none', 'sum', 'mean', 'max', and 'min'.

    Returns:
        Cross entropy loss of the input.
    """
    input1 = input1.to(tl.float32)
    input2 = input2.to(tl.float32)

    loss = -tl.sum(input2 * input1[tl.arange(input1.shape[0]), input2])

    if reduction == 'sum':
        return tl.sum(loss)
    elif reduction == 'mean':
        return tl.sum(loss) / tl.sum(tl.ones_like(loss))
    elif reduction == 'max':
        return tl.max(loss)
    elif reduction == 'min':
        return tl.min(loss)
    else:
        return loss
<|system|>
