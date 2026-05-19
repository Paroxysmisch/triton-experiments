_mean: Previous mean statistic to update.
        prev_var: Previous variance (M2) statistic to update.
        curr_count: Current count of input.
        mask: Mask indicating which elements should be included in the updates.
            The mask must be of the same shape as the input.

    Returns:
        Count, mean, and variance (M2) updated by Welford's algorithm.
    """
    input = input.to(tl.float32)

    count = prev_count + curr_count
    delta = input - prev_mean
    new_mean = prev_mean + delta * curr_count / count
    m2 = prev_var * (prev_count) + delta * (input - new_mean) * curr_count

    mask = mask.to(tl.float32)
    count = count * mask
    new_mean = (prev_mean * (prev_count * mask) + input * curr_count) / count
    m2 = (prev_var * (prev_count * mask) + delta * (input - new_mean) * curr_count) * mask

    return count, new_mean, m2

@triton.jit
def update_exp_mov_avg(input, prev, decay, count, mask: tl.constexpr):
    """
    Updates an exponential moving average.

    Args:
        input: Input used to update the exponential moving average.
            The input must be of the same shape as the mask.
        prev: Previous exponential moving average to update.
        decay: Decay rate of the exponential moving average.
        count: Count of input.
        mask: Mask indicating which elements should be included in the updates.
            The mask must be of the same shape as the input.

    Returns:
        Exponential moving average updated by Welford's algorithm.
    """
    input = input.to(tl.float32)

    mask = mask.to(tl.float32)
    decay = tl.where(count > 0, decay, 0)
    next_avg = (1 - decay) * input + decay * prev
    next_count = count * mask

    return next_avg, next_count

@triton.jit
def standardize(input, mean, inv_std, mean_inv_std_broadcasted, mask: tl.constexpr):
    """
    Standardizes the input.

    Args:
        input: Input to standardize.
            The input must be of the same shape as the mask.
        mean: Mean to subtract from the input.
            The mean must be of the same shape as the mask.
        inv_std: Inverse standard deviation to multiply the input by.
            The inverse standard deviation must be of the same shape as the mask.
        mean_inv_std_broadcasted: Flag for indicating if mean and inverse standard deviation
            are broadcasted to match the input.
            If true, both the mean and inverse standard deviation must be scalars.
            If false, both the mean and inverse standard deviation must be of the same shape as the input.
        mask: Mask indicating which elements should be included in the standardization.
            The mask must be of the same shape as the input.

    Returns:
        Standardized input.
    """
    input = input.to(tl.float32)

    if mean_inv_std_broadcasted:
        output = (input - mean) * inv_std

    else:
        mean = mean.to(tl.float32)
        inv_std = inv_std.to(tl.float32)
        mask = mask.to(tl.float32)
        output = (input - mean) * inv_std * mask

    return output

@triton.jit
def calc_norm(input, order: tl.constexpr, mask: tl.constexpr):
    """
    Calculates the L1 or L2 norm of the input.

    Args:
        input: Input whose norm is calculated.
            The input must be of the same shape as the mask.
        order: Order of the norm to calculate.
            Options are 1 for L1 norm and 2 for L2 norm.
        mask: Mask indicating which elements should be included in the norm calculation.
            The mask must be of the same shape as the input.

    Returns:
        L1 or L2 norm of the input.
    """
    input = input.to(tl.float32)

    if order == 1:
        output = tl.sum(tl.abs(input) * mask)

    elif order == 2:
        output = tl.sqrt(tl.sum((input * input) * mask))

    else:
        raise ValueError(f"Invalid order: {order}")

    return output

@triton.jit
def calc_neg_log_likelihood(input, target, ignore_index, log_input: tl.constexpr):
    """
    Calculates the negative log likelihood of the input given the target.

    Args:
        input: Input of the model given the target.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target the model is expected to predict.
            The target must be of shape [BLOCK_SIZE1].
        ignore_index: Target value to ignore in the calculation of the negative log likelihood.
        log_input: Flag for indicating if the log of the input should be taken.

    Returns:
        Negative log likelihood of the input given the target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.int32)

    if ignore_index >= 0:
        mask = target != ignore_index
    else:
        mask = target < 0 or target >= input.shape[1]

    target = target * mask
    input = tl.where(mask, input, 0)

    if log_input:
        input = tl.log(input)

    input = tl.max(input, axis=1)

    return -tl.sum(tl.gather(input, target, axis=1, index_dtype=tl.int32))

@triton.jit
def calc_cross_entropy(input, target, ignore_index, log_input: tl.constexpr):
    """
    Calculates the cross entropy between the input and the target.

    Args:
        input: Input of the model given the target.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target the model is expected to predict.
            The target must be of shape [BLOCK_SIZE1].
        ignore_index: Target value to ignore in the calculation of the cross entropy.
        log_input: Flag for indicating if the log of the input should be taken.

    Returns:
        Cross entropy between the input and the target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.int32)

    if ignore_index >= 0:
        mask = target != ignore_index
    else:
        mask = target < 0 or target >= input.shape[1]

    target = target * mask
    input = tl.where(mask, input, 0)

    if log_input:
        input = tl.log(input)

    input = -tl.gather(input, target, axis=1, index_dtype=tl.int32)

    return tl.sum(input)

@triton.jit
def relu_batch_norm_conv2d(input, weight, bias, stride, padding, dilation, groups, running_mean, running_var, bn_weight, bn_bias, training, momentum, eps, inplace: tl.constexpr):
    """
    Applies a 2D convolution over the input tensor, followed by batch normalization and then applies the ReLU activation function element-wise to the normalized result.

    Args:
        input: The input tensor to the layer.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2, BLOCK_SIZE3, BLOCK_SIZE4].
        weight: The learnable weights of the layer.
            The weights must be of shape [BLOCK_SIZE2, BLOCK_SIZE5, BLOCK_SIZE6, BLOCK_SIZE7].
        bias: The learnable bias of the layer.
            The bias must
