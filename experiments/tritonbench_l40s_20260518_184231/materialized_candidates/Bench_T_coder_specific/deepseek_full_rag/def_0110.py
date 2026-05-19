, reduction: tl.constexpr):
    """
    Calculates the per-example loss given an input and a target.

    Args:
        input: Input from which to calculate the loss.
            The input must be of shape [BLOCK_SIZE].
        target: Target of the input.
            The target must be of shape [BLOCK_SIZE].
        size: Size of the last dimension of the input.
        reduction: Reduction method used for calculating the loss.
            Options are 'none', 'mean', and 'sum'.

    Returns:
        Per-example loss.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    loss = -input * target + tl.log(1 + tl.exp(-tl.abs(input - target)))

    if reduction == "mean":
        loss = loss / size

    elif reduction == "sum":
        loss = loss.sum()

    return loss

@triton.jit
def calc_nll_loss(input, target, ignore_index, weight, size, reduction: tl.constexpr):
    """
    Calculates the negative log likelihood loss given an input and a target.

    Args:
        input: Input from which to calculate the loss.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target of the input.
            The target must be of shape [BLOCK_SIZE1].
        ignore_index: Index to ignore in the calculation of the loss.
        weight: Weight multiplied by the loss.
        size: Size of the last dimension of the input.
        reduction: Reduction method used for calculating the loss.
            Options are 'none', 'mean', and 'sum'.

    Returns:
        Negative log likelihood loss.
    """
    input = input.to(tl.float32)
    target = target.to(tl.int32)
    weight = weight.to(tl.float32)

    target_mask = target != ignore_index
    target = tl.where(target_mask, target, 0)
    loss = -weight * tl.sum(tl.where(target_mask, input[0, target], 0)) / size

    if reduction == "mean":
        loss = loss

    elif reduction == "sum":
        loss = loss * size

    return loss

@triton.jit
def calc_cross_entropy_loss(input, target, ignore_index, weight, size, reduction: tl.constexpr):
    """
    Calculates the cross entropy loss given an input and a target.

    Args:
        input: Input from which to calculate the loss.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target of the input.
            The target must be of shape [BLOCK_SIZE1].
        ignore_index: Index to ignore in the calculation of the loss.
        weight: Weight multiplied by the loss.
        size: Size of the last dimension of the input.
        reduction: Reduction method used for calculating the loss.
            Options are 'none', 'mean', and 'sum'.

    Returns:
        Cross entropy loss.
    """
    input = input.to(tl.float32)
    target = target.to(tl.int32)
    weight = weight.to(tl.float32)

    target_mask = target != ignore_index
    target = tl.where(target_mask, target, 0)
    input = tl.where(tl.arange(0, input.shape[1])[None, :] == target[:, None], input, 0.)
    loss = -weight * tl.sum(input) / size

    if reduction == "mean":
        loss = loss

    elif reduction == "sum":
        loss = loss * size

    return loss

@triton.jit
def calc_l1l2_loss(input, target, reduction_strategy, p, weight, size, reduction: tl.constexpr):
    """
    Calculates the L1 or L2 loss given an input and a target.

    Args:
        input: Input from which to calculate the loss.
            The input must be of shape [BLOCK_SIZE].
        target: Target of the input.
            The target must be of shape [BLOCK_SIZE].
        reduction_strategy: Reduction strategy used for calculating the loss.
            Options are 'none', 'mean', 'sum', 'mean_and_sum', and 'batch_mean'.
        p: Order of the norm used to calculate the loss.
        weight: Weight multiplied by the loss.
        size: Size of the last dimension of the input.
        reduction: Reduction method used for calculating the loss.
            Options are 'none', 'mean', and 'sum'.

    Returns:
        L1 or L2 loss.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)
    weight = weight.to(tl.float32)

    diff = input - target
    abs_diff = tl.abs(diff)
    square_diff = diff * diff

    if p == 0:
        loss = weight * tl.sum(abs_diff != 0) / size

    elif p == 1:
        loss = weight * tl.sum(abs_diff) / size

    elif p == 2:
        loss = weight * tl.sum(square_diff) / size

    else:
        loss = weight * tl.sum(tl.pow(square_diff, p)) / size

    if reduction == "mean":
        loss = loss

    elif reduction == "sum":
        loss = loss * size

    return loss

@triton.jit
def calc_mean(input, dim, keepdim, out_dtype, out_shape, mask: tl.constexpr):
    """
    Calculates the mean of the input along the specified dimension.

    Args:
        input: Input whose mean is calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        dim: Dimension along which to calculate the mean.
        keepdim: Flag for indicating whether the output tensor should have
            the same number of dimensions as the input, but with the
            specified dimension removed (if keepdim is True).
        out_dtype: Data type of the output tensor.
        out_shape: Shape of the output tensor.
            The shape must be of shape [BLOCK_SIZE3].
        mask: Mask indicating which elements should be included in the mean.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Mean of the input.
    """
    input = input.to(tl.float32)

    input = tl.where(mask[None, :], input, 0)
    mean = tl.sum(input, axis=dim)

    if not keepdim:
        out_shape = tl.inc(out_shape, dim)
        mean = tl.reshape(mean, out_shape)

    return mean.to(out_dtype)

@triton.jit
def exp_mean(input, dim, keepdim, dtype, out: tl.constexpr):
    """
    Applies the exponential function to each element in the input tensor
    and then computes the mean value of the result along the specified dimension
    or over all elements if no dimension is specified.

    Args:
        input: Input tensor.
        dim: Dimension or list of dimensions along which the mean is computed.
        keepdim: Flag for indicating whether the output tensor has
            the same number of dimensions as the input tensor,
            but with the specified dimension(s) removed if set to True.
        dtype: Data type of the output tensor.
        out: Optional output tensor.

    Returns:
        Tensor: The exponential mean of the input tensor.
    """
    input = input.to(tl.float32)

    out = tl.exp(input)
    out = calc_mean(out, dim, keepdim, dtype, out.shape, None)

    return out
