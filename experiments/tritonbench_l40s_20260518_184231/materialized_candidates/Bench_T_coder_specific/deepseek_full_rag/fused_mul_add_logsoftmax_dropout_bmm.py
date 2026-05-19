= prev_count + curr_count
    delta = input - prev_mean
    new_mean = prev_mean + delta * curr_count / count
    m2_delta = input - new_mean
    new_var = prev_var + tl.where(mask, m2_delta * delta, 0)

    return count, new_mean, new_var

@triton.jit
def ema_update(input, prev_ema, decay, last_dim_mask: tl.constexpr):
    """
    Updates exponential moving average.

    Args:
        input: Input used to update exponential moving average.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        prev_ema: Previous exponential moving average to update.
            The ema must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        decay: Decay rate of the exponential moving average.
        last_dim_mask: Mask for the last dimension indicating
            which elements should be updated.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Updated exponential moving average.
    """
    input = input.to(tl.float32)

    return tl.where(last_dim_mask[None, :], input, decay * prev_ema) * decay + prev_ema

@triton.jit
def normalize_input(input, mean, inv_std, last_dim_mask: tl.constexpr):
    """
    Standardizes the input using mean and inverse standard deviation.

    Args:
        input: Input to standardize.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        mean: Mean to subtract from the input.
            The mean must be of shape [BLOCK_SIZE1].
        inv_std: Inverse standard deviation to multiply the input by.
            The inv_std must be of shape [BLOCK_SIZE1].
        last_dim_mask: Mask for the last dimension indicating
            which elements should be standardized.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Standardized input.
    """
    input = input.to(tl.float32)

    return tl.where(last_dim_mask[None, :], (input - mean[:, None]) * inv_std[:, None], 0)

@triton.jit
def calc_l1l2_norm(input, order: tl.constexpr, last_dim_mask: tl.constexpr):
    """
    Calculates the L1 or L2 norm of the input along the last dimension.

    Args:
        input: Input whose L1 or L2 norm is calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        order: Order of the norm to calculate.
            Options are 1 for L1 norm and 2 for L2 norm.
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the calculations.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        L1 or L2 norm of the input.
    """
    input = input.to(tl.float32)

    if order == 1:
        norm = tl.sum(tl.abs(input), axis=1)

    else:
        diff = tl.where(last_dim_mask[None, :], input, 0)
        norm = tl.sqrt(tl.sum(diff * diff, axis=1))

    return norm

@triton.jit
def calc_neg_log_likelihood(input, target, last_dim_mask: tl.constexpr):
    """
    Calculates the negative log likelihood of the input given the target.

    Args:
        input: Input (log probabilities) to calculate the negative log likelihood of.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target classes.
            The target must be of shape [BLOCK_SIZE1].
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the calculations.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Negative log likelihood of the input given the target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.int32)

    target_masked = tl.where(last_dim_mask, target, -1)
    nll = tl.log(tl.sum(tl.exp(input), axis=1)) - tl.take_along_axis(input, target[:, None], axis=1)[:, 0]
    nll = tl.where(target_masked >= 0, nll, 0)

    return nll

@triton.jit
def calc_cross_entropy(input, target, last_dim_mask: tl.constexpr):
    """
    Calculates the cross entropy loss of the input given the target.

    Args:
        input: Input (log probabilities) to calculate the cross entropy loss of.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target classes.
            The target must be of shape [BLOCK_SIZE1].
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the calculations.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Cross entropy loss of the input given the target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.int32)

    target_masked = tl.where(last_dim_mask, target, -1)
    ce = tl.sum(tl.take_along_axis(input, target[:, None], axis=1)[:, 0]) / -tl.sum(tl.exp(input), axis=1)
    ce = tl.where(target_masked >= 0, ce, 0)

    return ce

@triton.jit
def fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, *, out=None):
    """
    Performs a fused operation combining element-wise multiplication, addition,
    log-softmax activation, dropout, and batch matrix multiplication.

    Args:
        input1: First input tensor for element-wise multiplication.
            The tensor can be of any shape.
        input2: Second input tensor for element-wise multiplication.
            The tensor must be broadcastable to the shape of input1.
        other: Tensor or scalar to be added to the result of the element-wise
            multiplication. The tensor must be broadcastable to the shape of
            the result of the element-wise multiplication.
        mat2: Second input tensor for batch matrix multiplication.
            The tensor must be of shape [B, D_in, D_out] if the dropout output
            has shape [B, N, D_in].
        p: Probability of an element being zeroed.
            Default is 0.5.
        training: Flag for indicating training mode.
            If True, dropout is applied, and if False, dropout is not applied.
            Default is True.
        inplace: Flag for indicating if the input tensors should be modified in-place.
            If True, input1 and input2 will be modified, and if False,
            input1 and input2 will not be modified.
            Default is False.
        dim: Dimension to apply the log-softmax activation to.
            Default is -1.
        out: Preallocated output tensor.
            The output tensor must be of the same shape as the result of the
            element-wise multiplication.

    Returns:
        Result of the fused operation.
    """
    if inplace:
        input1 *= input2
    else:
        input1 = input1 * input2

    input1 += other
    input1 = glu(input1, input2, None, "relu")
    input1 = tl.log(input1)

    if training:
        input1 = tl.nn.dropout(input1, p)

    if out is None:
        out = tl.zeros(input1.shape,
