"""
    input = input.to(tl.float32)

    return (input - mean) * inv_std * weight + bias

@triton.jit
def calc_l1l2_norm(input, ord: tl.constexpr, dim, keepdim: tl.constexpr):
    """
    Calculates the L1 or L2 norm of the input along a specific dimension.

    Args:
        input: Input whose L1 or L2 norm is calculated.
        ord: Order of the norm.
            Options are 1 for L1 norm and 2 for L2 norm.
        dim: Dimension along which the norm is calculated.
        keepdim: Flag for keeping the dimension in the output.

    Returns:
        L1 or L2 norm of the input.
    """
    if ord == 1:
        return tl.sum(tl.abs(input), axis=dim, keepdim=keepdim)

    elif ord == 2:
        return tl.sqrt(tl.sum(input * input, axis=dim, keepdim=keepdim))

    else:
        return tl.sum(tl.abs(input) ** ord, axis=dim, keepdim=keepdim) / ord

@triton.jit
def calc_nll_loss(input, target, ignore_index, last_dim_mask: tl.constexpr):
    """
    Calculates the negative log likelihood loss between the input and target.

    Args:
        input: Input to take the negative log likelihood of.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target of the input.
            The target must be of shape [BLOCK_SIZE1] and
            values must be in the range [0, BLOCK_SIZE2 - 1].
        ignore_index: Target value to ignore.
            If target is equal to ignore_index, the corresponding input
            element does not contribute to the loss.
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the calculations.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Negative log likelihood loss between the input and target.
    """
    input = input.to(tl.float32)

    log_probs = input - tl.logsumexp(input, axis=1)[:, None]
    nll_loss = -tl.sum(tl.where(last_dim_mask, log_probs[range(target.shape[0]), target], 0)) / target.shape[0]
    nll_loss += tl.sum(tl.where(target == ignore_index, 0, log_probs[range(target.shape[0]), target])) / target.shape[0]

    return nll_loss

@triton.jit
def calc_cross_entropy_loss(input, target, weight, ignore_index, last_dim_mask: tl.constexpr):
    """
    Calculates the cross entropy loss between the input and target.

    Args:
        input: Input to take the cross entropy of.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        target: Target of the input.
            The target must be of shape [BLOCK_SIZE1] and
            values must be in the range [0, BLOCK_SIZE2 - 1].
        weight: Weight multiplied by the input.
        ignore_index: Target value to ignore.
            If target is equal to ignore_index, the corresponding input
            element does not contribute to the loss.
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the calculations.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Cross entropy loss between the input and target.
    """
    input = input.to(tl.float32)

    log_probs = input - tl.logsumexp(input, axis=1)[:, None]
    loss = tl.sum(tl.where(last_dim_mask, log_probs[range(target.shape[0]), target], 0)) / target.shape[0]
    loss -= tl.sum(tl.where(target == ignore_index, 0, weight * log_probs[range(target.shape[0]), target])) / target.shape[0]

    return loss

@triton.jit
def matmul_accum(a, b, c, accumulate, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Accumulates matrix multiplications of input tensors.

    Args:
        a: First operand of matrix multiplication.
            The operand must be of shape [BLOCK_M, BLOCK_K].
        b: Second operand of matrix multiplication.
            The operand must be of shape [BLOCK_K, BLOCK_N].
        c: Result of matrix multiplication to accumulate.
            The result must be of shape [BLOCK_M, BLOCK_N].
        accumulate: Flag for indicating whether the result should be accumulated.

    Returns:
        Result of matrix multiplication of the operands.
    """
    if accumulate:
        return tl.dot(a, b) + c
    else:
        return tl.dot(a, b)

@triton.jit
def matmul_relu(a, b, c, accumulate, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Accumulates matrix multiplications of input tensors.

    Args:
        a: First operand of matrix multiplication.
            The operand must be of shape [BLOCK_M, BLOCK_K].
        b: Second operand of matrix multiplication.
            The operand must be of shape [BLOCK_K, BLOCK_N].
        c: Result of matrix multiplication to accumulate.
            The result must be of shape [BLOCK_M, BLOCK_N].
        accumulate: Flag for indicating whether the result should be accumulated.

    Returns:
        Result of matrix multiplication of the operands.
    """
    if accumulate:
        return tl.dot(a, b) + c
    else:
        return tl.dot(a, b)

@triton.jit
def matmul_relu6(a, b, c, accumulate, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Accumulates matrix multiplications of input tensors.

    Args:
        a: First operand of matrix multiplication.
            The operand must be of shape [BLOCK_M, BLOCK_K].
        b: Second operand of matrix multiplication.
            The operand must be of shape [BLOCK_K, BLOCK_N].
        c: Result of matrix multiplication to accumulate.
            The result must be of shape [BLOCK_M, BLOCK_N].
        accumulate: Flag for indicating whether the result should be accumulated.

    Returns:
        Result of matrix multiplication of the operands.
    """
    if accumulate:
        return tl.dot(a, b) + c
    else:
        return tl.dot(a, b)

@triton.jit
def matmul_sigmoid(a, b, c, accumulate, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Accumulates matrix multiplications of input tensors.

    Args:
        a: First operand of matrix multiplication.
            The operand must be of shape [BLOCK_M, BLOCK_K].
        b: Second operand of matrix multiplication.
            The operand must be of shape [BLOCK_K, BLOCK_N].
        c: Result of matrix multiplication to accumulate.
            The result must be of shape [BLOCK_M, BLOCK_N].
        accumulate: Flag for indicating whether the result should be accumulated.

    Returns:
        Result of matrix multiplication of the operands.
    """
    if accumulate:
        return tl.dot(a, b) + c
    else:
        return tl.dot(a
