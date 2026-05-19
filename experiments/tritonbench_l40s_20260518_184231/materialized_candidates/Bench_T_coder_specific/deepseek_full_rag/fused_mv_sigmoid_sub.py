+ momentum * new_val

@triton.jit
def standardize(input, mean, inv_std, eps, last_dim_mask: tl.constexpr):
    """
    Standardizes the input using mean and inverse standard deviation.

    Args:
        input: Input to standardize.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        mean: Mean of the input.
            The mean must be of shape [BLOCK_SIZE1].
        inv_std: Inverse standard deviation of the input.
            The inverse standard deviation must be of shape [BLOCK_SIZE1].
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the calculations.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Standardized input.
    """
    input = input.to(tl.float32)

    standardized = (input - mean[:, None]) * inv_std[:, None]
    output = standardized * tl.rsqrt(inv_std ** 2 + eps)

    return tl.where(last_dim_mask[None, :], output, 0.)

@triton.jit
def l1_l2_loss(input, target, reduction: tl.constexpr, weight: tl.constexpr, l1: tl.constexpr):
    """
    Calculates L1/L2 loss between the input and target.

    Args:
        input: Input tensor.
            The input must be of the same shape as the target.
        target: Target tensor.
            The target must be of the same shape as the input.
        reduction: Reduction strategy for the output.
            Options are 'none', 'mean', and 'sum'.
        weight: Weighting factor for the loss.
        l1: Flag for indicating if L1 loss should be calculated instead of L2.

    Returns:
        L1/L2 loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    diff = input - target

    if l1:
        loss = weight * tl.sum(tl.abs(diff))

    else:
        loss = weight * tl.sum(diff * diff)

    if reduction == "mean":
        loss /= tl.numel(input)

    elif reduction == "sum":
        loss = tl.sum(loss)

    return loss

@triton.jit
def nll_loss(input, target, reduction: tl.constexpr, weight, ignore_index: tl.constexpr):
    """
    Calculates negative log likelihood loss between the input and target.

    Args:
        input: Input tensor.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target tensor.
            The target must be of shape [BLOCK_SIZE1].
        reduction: Reduction strategy for the output.
            Options are 'none', 'mean', and 'sum'.
        weight: Weighting factor for the loss.
            The weight must be of shape [BLOCK_SIZE2].
        ignore_index: Target value to ignore.

    Returns:
        Negative log likelihood loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.int32)
    weight = weight.to(tl.float32)

    loss = weight * tl.where(target != ignore_index, -input[range(target.shape[0]), target], 0.)

    if reduction == "mean":
        loss /= tl.sum(weight)

    elif reduction == "sum":
        loss = tl.sum(loss)

    return loss

@triton.jit
def cross_entropy_loss(input, target, reduction: tl.constexpr, weight, ignore_index: tl.constexpr):
    """
    Calculates cross entropy loss between the input and target.

    Args:
        input: Input tensor.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target tensor.
            The target must be of shape [BLOCK_SIZE1].
        reduction: Reduction strategy for the output.
            Options are 'none', 'mean', and 'sum'.
        weight: Weighting factor for the loss.
            The weight must be of shape [BLOCK_SIZE2].
        ignore_index: Target value to ignore.

    Returns:
        Cross entropy loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.int32)
    weight = weight.to(tl.float32)

    input = input - tl.max(input, axis=1)[:, None]
    loss = weight * tl.where(target != ignore_index, -input[range(target.shape[0]), target], 0.)

    if reduction == "mean":
        loss /= tl.sum(weight)

    elif reduction == "sum":
        loss = tl.sum(loss)

    return loss

@triton.jit
def norm(input, p: tl.constexpr, dim, keepdim: tl.constexpr):
    """
    Calculates the p-norm of the input along the specified dimension.

    Args:
        input: Input tensor.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        p: Order of the norm.
        dim: Dimension along which the norm is calculated.
        keepdim: Flag for keeping the dimension of the input in the output.

    Returns:
        p-norm of the input along the specified dimension.
    """
    input = input.to(tl.float32)

    if p == 0:
        output = tl.max(tl.abs(input), axis=dim, keepdim=keepdim)

    elif p == 1:
        output = tl.sum(tl.abs(input), axis=dim, keepdim=keepdim)

    elif p == 2:
        output = tl.sqrt(tl.sum(input * input, axis=dim, keepdim=keepdim))

    elif p == tl.inf:
        output = tl.max(tl.abs(input), axis=dim, keepdim=keepdim)

    else:
        output = tl.sum(tl.pow(tl.abs(input), p), axis=dim, keepdim=keepdim)
        output = tl.pow(output, 1 / p)

    return output
