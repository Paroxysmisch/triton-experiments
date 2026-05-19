reduction: tl.constexpr):
    """
    Calculates the probability distribution loss.

    Args:
        input: Input to loss function.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target of the loss function.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        size: Total size of the input.
        reduction: Reduction type.
            Options are 'sum', 'mean', and 'none'.

    Returns:
        Probability distribution loss.
    """
    input = input.to(tl.float32)

    loss = -target * tl.log(input)

    if reduction == 'sum':
        return tl.sum(loss)

    elif reduction == 'mean':
        return tl.sum(loss) / size

    else:
        return loss

@triton.jit
def calc_cross_entropy(input, target, size, reduction: tl.constexpr):
    """
    Calculates the cross entropy loss.

    Args:
        input: Input to loss function.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target of the loss function.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        size: Total size of the input.
        reduction: Reduction type.
            Options are 'sum', 'mean', and 'none'.

    Returns:
        Cross entropy loss.
    """
    input = input.to(tl.float32)

    loss = target * tl.log(input)

    if reduction == 'sum':
        return tl.sum(loss)

    elif reduction == 'mean':
        return tl.sum(loss) / size

    else:
        return loss

@triton.jit
def calc_l1_loss(input, target, reduction: tl.constexpr):
    """
    Calculates the L1 norm loss.

    Args:
        input: Input to loss function.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target of the loss function.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        reduction: Reduction type.
            Options are 'sum', 'mean', and 'none'.

    Returns:
        L1 norm loss.
    """
    input = input.to(tl.float32)

    loss = tl.sum(tl.abs(input - target))

    if reduction == 'sum':
        return loss

    elif reduction == 'mean':
        return loss / tl.numel(input)

    else:
        return loss

@triton.jit
def calc_l2_loss(input, target, reduction: tl.constexpr):
    """
    Calculates the L2 norm loss.

    Args:
        input: Input to loss function.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target of the loss function.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        reduction: Reduction type.
            Options are 'sum', 'mean', and 'none'.

    Returns:
        L2 norm loss.
    """
    input = input.to(tl.float32)

    loss = tl.sum((input - target) ** 2)

    if reduction == 'sum':
        return loss

    elif reduction == 'mean':
        return loss / tl.numel(input)

    else:
        return loss
<|endoftext|>
