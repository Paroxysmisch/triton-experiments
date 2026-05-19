l.constexpr):
    """
    Updates the mean and variance using Welford's algorithm.

    Args:
        input: Input whose mean and variance are calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        prev_count: Count of previous inputs.
        prev_mean: Mean of previous inputs.
            The mean must be of shape [BLOCK_SIZE2].
        prev_var: Variance of previous inputs.
            The variance must be of shape [BLOCK_SIZE2].
        curr_count: Count of current inputs.
        mask: Mask for the last dimension indicating
            which elements should be included in the calculations.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Mean and variance of the input.
    """
    input = input.to(tl.float32)

    curr_mean = tl.where(mask[None, :], tl.sum(input, axis=0) / curr_count, 0)
    diff = tl.where(mask[None, :], input - prev_mean[None, :], 0)
    curr_var = tl.sum(diff * diff, axis=0) / curr_count

    return curr_mean, curr_var

@triton.jit
def update_ema(input, prev_mean, prev_count, alpha):
    """
    Updates the mean using exponential moving average.

    Args:
        input: Input whose mean is calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        prev_mean: Mean of previous inputs.
            The mean must be of shape [BLOCK_SIZE2].
        prev_count: Count of previous inputs.
        alpha: Smoothing factor.

    Returns:
        Mean of the input.
    """
    input = input.to(tl.float32)

    curr_mean = alpha * tl.sum(input, axis=0) / prev_count + (1 - alpha) * prev_mean

    return curr_mean

@triton.jit
def standardize(input, mean, inv_std):
    """
    Standardizes the input by subtracting the mean and dividing by the inverse standard deviation.

    Args:
        input: Input to standardize.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        mean: Mean of the input.
            The mean must be of shape [BLOCK_SIZE2].
        inv_std: Inverse standard deviation of the input.
            The inverse standard deviation must be of shape [BLOCK_SIZE2].

    Returns:
        Standardized input.
    """
    input = input.to(tl.float32)

    return (input - mean[:, None]) * inv_std[:, None]

@triton.jit
def calc_l1l2_norm_loss(input1, input2, p: tl.constexpr, reduction: tl.constexpr):
    """
    Calculates the L1 or L2 norm loss between two inputs.

    Args:
        input1: First input.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        input2: Second input.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        p: Order of the norm.
            Options are 1 for L1 norm and 2 for L2 norm.
        reduction: Type of reduction to apply to the loss.
            Options are 'mean' for average loss, 'sum' for sum of loss,
            and 'none' for no reduction.

    Returns:
        L1 or L2 norm loss between the two inputs.
    """
    input1 = input1.to(tl.float32)
    input2 = input2.to(tl.float32)

    diff = input1 - input2
    loss = tl.sum(tl.pow(tl.abs(diff), p))

    if reduction == 'mean':
        loss = loss / tl.size(input1)
    elif reduction == 'sum':
        pass
    elif reduction == 'none':
        loss = loss
    else:
        raise ValueError(f"Unknown reduction type: {reduction}")

    return loss

@triton.jit
def calc_nll_loss(input, target, reduction: tl.constexpr):
    """
    Calculates the negative log likelihood loss between the input and target.

    Args:
        input: Input to calculate the loss for.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target to calculate the loss against.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        reduction: Type of reduction to apply to the loss.
            Options are 'mean' for average loss, 'sum' for sum of loss,
            and 'none' for no reduction.

    Returns:
        Negative log likelihood loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    loss = -tl.sum(target * tl.log(input))

    if reduction == 'mean':
        loss = loss / tl.size(input)
    elif reduction == 'sum':
        pass
    elif reduction == 'none':
        loss = loss
    else:
        raise ValueError(f"Unknown reduction type: {reduction}")

    return loss

@triton.jit
def calc_cross_entropy_loss(input, target, reduction: tl.constexpr):
    """
    Calculates the cross entropy loss between the input and target.

    Args:
        input: Input to calculate the loss for.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target to calculate the loss against.
            The target must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        reduction: Type of reduction to apply to the loss.
            Options are 'mean' for average loss, 'sum' for sum of loss,
            and 'none' for no reduction.

    Returns:
        Cross entropy loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    loss = tl.sum(target * tl.log(input))

    if reduction == 'mean':
        loss = loss / tl.size(input)
    elif reduction == 'sum':
        pass
    elif reduction == 'none':
        loss = loss
    else:
        raise ValueError(f"Unknown reduction type: {reduction}")

    return loss

@triton.jit
def apply_act_func(input, param, act_func: tl.constexpr, log: tl.constexpr, fp16: tl.constexpr, tf32: tl.constexpr):
    """
    Applies an activation function to the input.

    Args:
        input: Input to apply the activation function to.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        log: Flag for indicating if the log of activation function should be taken.
        fp16: Flag for converting operands to FP16.
        tf32: Flag for performing matrix multiplication in TF32.

    Returns:
        Input transformed by the activation function.
    """
    if fp16:
        input = input.to(tl.float16)

    if act_func == 'sigmoid':
        output = tl.sigmoid(input)
    elif act_func == 'tanh':
        output = tl.tanh(input)
    elif act_func == 'relu':
        output = tl.relu(input)
    elif act_func == 'gelu':
        output = tl.gelu(input)
    elif act_
