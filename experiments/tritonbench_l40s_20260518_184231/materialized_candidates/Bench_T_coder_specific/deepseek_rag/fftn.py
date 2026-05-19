, decay, mask: tl.constexpr):
    """
    Updates an exponential moving average.

    Args:
        prev_ema: Previous exponential moving average to update.
            The ema must be of the same shape as the mask.
        new_val: New value to incorporate into the exponential moving average.
            The new value must be of the same shape as the mask.
        decay: Decay factor for the exponential moving average.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of the same shape as the ema.

    Returns:
        Updated exponential moving average.
    """
    new_val = new_val.to(tl.float32)

    ema = prev_ema + decay * (new_val - prev_ema)
    ema = tl.where(mask, new_val, ema)

    return ema

@triton.jit
def standardize(input, mean, inv_std):
    """
    Standardizes the input using mean and inverse standard deviation.

    Args:
        input: Input to standardize.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        mean: Mean of the input.
            The mean must be of shape [BLOCK_SIZE1].
        inv_std: Inverse standard deviation of the input.
            The inv_std must be of shape [BLOCK_SIZE1].

    Returns:
        Input standardized using mean and inverse standard deviation.
    """
    input = input.to(tl.float32)

    return (input - mean[:, None]) * inv_std[:, None]

@triton.jit
def calc_l1_l2_loss(input1, input2, p: tl.constexpr):
    """
    Calculates the L1 or L2 loss between two inputs.

    Args:
        input1: First input.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        input2: Second input.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        p: Order of the norm to use for the loss calculation.
            Options are 1 for L1 loss and 2 for L2 loss.

    Returns:
        L1 or L2 loss between the two inputs.
    """
    input1 = input1.to(tl.float32)
    input2 = input2.to(tl.float32)

    return tl.sum(tl.pow(tl.abs(input1 - input2), p))

@triton.jit
def calc_neg_log_like(input, target, log_probs):
    """
    Calculates the negative log likelihood loss between an input and a target.

    Args:
        input: Input to calculate the loss for.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target to calculate the loss against.
            The target must be of shape [BLOCK_SIZE1].
        log_probs: Log probabilities of the target classes.
            The log_probs must be of shape [BLOCK_SIZE2, BLOCK_SIZE3].

    Returns:
        Negative log likelihood loss between the input and the target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.int32)

    return -tl.sum(tl.gather(log_probs, target, axis=1) * input)

@triton.jit
def calc_cross_entropy(input, target, log_probs):
    """
    Calculates the cross entropy loss between an input and a target.

    Args:
        input: Input to calculate the loss for.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target to calculate the loss against.
            The target must be of shape [BLOCK_SIZE1].
        log_probs: Log probabilities of the target classes.
            The log_probs must be of shape [BLOCK_SIZE2, BLOCK_SIZE3].

    Returns:
        Cross entropy loss between the input and the target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.int32)

    return tl.sum(tl.gather(log_probs, target, axis=1) * input)

Document 2:
This document is a reference for how to use the Triton language in implementing mathematical operations on tensors.

@triton.jit
def accum_linear(accum, input1, input2, fp16: tl.constexpr, tf32: tl.constexpr):
    ...
    # Use the Triton language to implement the matrix multiplication accumulation
    if fp16:
        input1 = input1.to(tl.float16)
        input2 = input2.to(tl.float16)
    return accum + tl.dot(input1, input2, allow_tf32=tf32)

@triton.jit
def glu(input1, input2, param, act_func: tl.constexpr):
    ...
    # Use the Triton language to implement the gated linear unit application
    return input1 * apply_act_func(input2, None, None, None, param, act_func, False)

@triton.jit
def softmax(input, log: tl.constexpr):
    ...
    # Use the Triton language to implement the softmax normalization
    input = input.to(tl.float32)
    input = input - tl.max(input, axis=1)[:, None]
    numerator = tl.exp(input)
    denominator = tl.sum(numerator, axis=1)[:, None]
    if log:
        output = input - tl.log(denominator)
    else:
        output = numerator / denominator
    return output

@triton.jit
def calc_mean_and_inv_std(input, last_dim, eps, last_dim_mask: tl.constexpr):
    ...
    # Use the Triton language to implement the mean and inverse standard deviation calculation
    input = input.to(tl.float32)
    mean = tl.sum(input, axis=1) / last_dim
    diff = tl.where(last_dim_mask[None, :], input - mean[:, None], 0)
    inv_std = tl.rsqrt(tl.sum(diff * diff, axis=1) / last_dim + eps)
    return mean, inv_std

@triton.jit
def update_welford(input, prev_count, prev_mean, prev_var, curr_count, mask: tl.constexpr):
    ...
    # Use the Triton language to implement the Welford's algorithm update
    input = input.to(tl.float32)
    count = prev_count + curr_count
    mean = (tl.sum(input) - curr_count * prev_mean) / count
    deltas = tl.where(mask, (input - mean) * (input - prev_mean), 0.)
    var = prev_var + tl.sum(deltas)
    return count, mean, var

@triton.jit
def update_ema(prev_ema, new_val, decay, mask: tl.constexpr):
    ...
    # Use the Triton language to implement the exponential moving average update
    new_val = new_val.to(tl.float32)
    ema = prev_ema + decay * (new_val - prev_ema)
    ema = tl.where(mask, new_val, ema)
    return ema

@triton.jit
def standardize(input, mean, inv_std):
    ...
    # Use the Triton language to implement the input standardization
    input = input.to(tl.float32)
    return (input - mean[:, None]) * inv_std[:, None]

@triton.jit
def calc_l1_l2_loss(input1, input2, p: tl.constexpr):
    ...
    # Use the Triton language to implement the L1/L2 norm loss calculation
    input1 = input1.
