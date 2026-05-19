mean, inv_std, dim, eps, mask: tl.constexpr):
    """
    Standardizes the input using mean and inverse standard deviation.

    Args:
        input: Input to standardize.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        mean: Mean to subtract from the input.
            The mean must be of shape [BLOCK_SIZE1].
        inv_std: Inverse standard deviation to multiply the input by.
            The inverse standard deviation must be of shape [BLOCK_SIZE1].
        dim: Dimension of the input to reduce.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        mask: Mask for the input indicating which elements should be included
            in the mean and standard deviation calculations.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Standardized input.
    """
    input = input.to(tl.float32)

    input_centered = tl.where(mask[:, None], input - mean[None, :], 0.)
    norm = tl.sqrt(tl.sum(input_centered * input_centered, axis=dim) /
                   tl.sum(mask) + eps)
    output = input_centered * inv_std[None, :] / norm

    return output

@triton.jit
def l1_l2_norm_loss(input, target, dim, reduction_type: tl.constexpr, mask: tl.constexpr):
    """
    Calculates the L1 or L2 norm loss between the input and target.

    Args:
        input: Input tensor.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target tensor.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        dim: Dimension of the input to reduce.
        reduction_type: Type of reduction to perform.
            Options are 'mean' and 'sum'.
        mask: Mask for the input and target indicating which elements
            should be included in the loss calculation.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        L1 or L2 norm loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    diff = tl.where(mask[:, None], input - target, 0.)

    if reduction_type == 'mean':
        return tl.sum(diff * diff) / tl.sum(mask)
    elif reduction_type == 'sum':
        return tl.sum(diff * diff)

@triton.jit
def negative_log_likelihood_loss(input, target, dim, ignore_index: tl.constexpr, mask: tl.constexpr):
    """
    Calculates the negative log likelihood loss between the input and target.

    Args:
        input: Input tensor.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target tensor.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        dim: Dimension of the input to reduce.
        ignore_index: Index to ignore in the loss calculation.
        mask: Mask for the input and target indicating which elements
            should be included in the loss calculation.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Negative log likelihood loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    target_mask = target != ignore_index
    loss_mask = mask & target_mask

    input_clamped = tl.where(loss_mask, input, 0.)
    target_clamped = tl.where(loss_mask, target, 0.)

    return -tl.sum(target_clamped * tl.log(input_clamped)) / tl.sum(loss_mask)

@triton.jit
def cross_entropy_loss(input, target, dim, ignore_index: tl.constexpr, mask: tl.constexpr):
    """
    Calculates the cross entropy loss between the input and target.

    Args:
        input: Input tensor.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target tensor.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        dim: Dimension of the input to reduce.
        ignore_index: Index to ignore in the loss calculation.
        mask: Mask for the input and target indicating which elements
            should be included in the loss calculation.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Cross entropy loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    target_mask = target != ignore_index
    loss_mask = mask & target_mask

    input_clamped = tl.where(loss_mask, input, 0.)
    target_clamped = tl.where(loss_mask, target, 0.)

    return -tl.sum(target_clamped * tl.log(input_clamped)) / tl.sum(loss_mask)

@triton.jit
def std(input, dim=None, *, correction=1, keepdim=False, out=None) -> tl.Tensor:
    """
    Calculates the standard deviation over the specified dimensions of the input tensor.

    Args:
        input: The input tensor.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        dim: The dimension or dimensions to reduce.
            If None, the input is flattened, reducing the last dimension only.
        correction: Difference between the sample size and sample degrees of freedom.
            Defaults to Bessel's correction, correction=1.
        keepdim: Whether the output tensor has dim retained or not.
        out: The output tensor.

    Returns:
        The standard deviation of the input tensor.
    """
    input = input.to(tl.float32)

    if dim is None:
        input = input.ravel()
        dim = -1

    input_mean = tl.sum(input, axis=dim) / tl.numel(input)
    diff = input - input_mean
    sq_diff = diff * diff
    var = tl.sum(sq_diff, axis=dim) / tl.numel(sq_diff)
    std = tl.sqrt(var + correction)

    if not keepdim:
        std = tl.squeeze(std, dim)

    return std
