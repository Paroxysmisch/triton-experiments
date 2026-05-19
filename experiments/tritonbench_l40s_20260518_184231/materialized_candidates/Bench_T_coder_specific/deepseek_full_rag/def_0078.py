delta = tl.where(mask, input - mean, 0)
    var = prev_var + tl.sum(delta * delta)

    return count, mean, var

@triton.jit
def ema_update(prev_value, new_value, count, decay):
    """
    Updates value using exponential moving average update.

    Args:
        prev_value: Previous value to update.
        new_value: New value to incorporate into the update.
        count: Number of new values that will be averaged over.
        decay: Decay rate of the moving average.

    Returns:
        Updated value.
    """
    decay = tl.maximum(tl.minimum(decay, 1.0), 0.0)
    weight = tl.pow(1 - decay, count)
    return weight * prev_value + new_value

@triton.jit
def standardize(input, mean, inv_std, last_dim_mask: tl.constexpr):
    """
    Standardizes the input with a given mean and inverse standard deviation.

    Args:
        input: Input to standardize.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        mean: Mean to standardize the input with.
            The mean must be of shape [BLOCK_SIZE1].
        inv_std: Inverse standard deviation to standardize the input with.
            The inverse standard deviation must be of shape [BLOCK_SIZE1].
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the standardization.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Standardized input.
    """
    input = input.to(tl.float32)

    output = (input - mean[:, None]) * inv_std[:, None]
    output = tl.where(last_dim_mask[None, :], output, 0)

    return output

@triton.jit
def calc_l1l2_norm(input, p, last_dim_mask: tl.constexpr):
    """
    Calculates the L1 or L2 norm of the input along the last dimension.

    Args:
        input: Input whose L1 or L2 norm is calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        p: Order of the norm to calculate.
            Options are 1 for L1 norm and 2 for L2 norm.
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the norm calculation.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        L1 or L2 norm of the input.
    """
    input = input.to(tl.float32)

    output = tl.sum(tl.where(last_dim_mask[None, :], tl.abs(input) ** p, 0)) ** (1 / p)

    return output

@triton.jit
def calc_neg_log_likelihood(logits, labels, ignore_index, last_dim_mask: tl.constexpr):
    """
    Calculates the negative log likelihood loss given logits and labels.

    Args:
        logits: Logits from model.
            The logits must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        labels: Ground truth labels.
            The labels must be of shape [BLOCK_SIZE1].
        ignore_index: Target value to ignore.
            No contribution is made to the sum of the negative log likelihood
            for targets equal to the ignore index.
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the loss calculation.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Negative log likelihood loss.
    """
    logits = logits.to(tl.float32)
    labels = labels.to(tl.int32)

    logits = logits - tl.max(logits, axis=1)[:, None]
    logits = tl.where(last_dim_mask[None, :], logits, -float('inf'))
    likelihoods = tl.log(tl.sum(tl.exp(logits), axis=1))

    mask = labels != ignore_index
    nll = -likelihoods[mask] + logits[mask, labels[mask]]
    nll = tl.sum(nll) / tl.sum(mask.to(tl.float32))

    return nll

@triton.jit
def calc_cross_entropy(logits, labels, ignore_index, last_dim_mask: tl.constexpr):
    """
    Calculates the cross entropy loss given logits and labels.

    Args:
        logits: Logits from model.
            The logits must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        labels: Ground truth labels.
            The labels must be of shape [BLOCK_SIZE1].
        ignore_index: Target value to ignore.
            No contribution is made to the sum of the cross entropy
            for targets equal to the ignore index.
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the loss calculation.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Cross entropy loss.
    """
    logits = logits.to(tl.float32)
    labels = labels.to(tl.int32)

    logits = logits - tl.max(logits, axis=1)[:, None]
    probs = tl.exp(logits)
    logits = tl.where(last_dim_mask[None, :], logits, -float('inf'))
    likelihoods = tl.log(tl.sum(tl.exp(logits), axis=1))

    mask = labels != ignore_index
    ce = likelihoods[mask] - probs[mask, labels[mask]]
    ce = tl.sum(ce) / tl.sum(mask.to(tl.float32))

    return ce
