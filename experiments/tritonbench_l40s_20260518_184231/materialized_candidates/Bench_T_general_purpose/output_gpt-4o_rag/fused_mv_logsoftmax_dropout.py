import triton
import triton.language as tl

@triton.jit
def fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=0, out=None):
    """
    Performs matrix-vector multiplication, log-softmax activation, and dropout.

    Args:
        input: Matrix of shape [n, m] (Input matrix A).
        vec: Vector of shape [m] (Input vector v).
        p: Dropout probability. Defaults to 0.5.
        training: Whether to apply dropout during training. Defaults to True.
        inplace: Whether to apply the operation in-place. Defaults to False.
        dim: Dimension along which to compute the log-softmax. Should be 0 or -1.
        out: Optional output tensor.

    Returns:
        Tensor after performing the fused operations.
    """

    # Matrix-vector multiplication: z = A * v
    z = tl.dot(input, vec)

    # Log-softmax operation: s = log(exp(z) / sum(exp(z)))
    z_exp = tl.exp(z - tl.max(z))  # Stability trick to prevent overflow in exp
    sum_exp = tl.sum(z_exp, axis=dim, keepdims=True)
    s = z - tl.log(sum_exp)

    # Dropout operation (only during training)
    if training:
        # Generate dropout mask
        dropout_mask = tl.random.uniform(shape=s.shape, dtype=tl.float32) > p
        s = tl.where(dropout_mask, s, tl.zeros_like(s))
        s = s * (1.0 / (1.0 - p))  # Scale by 1/(1-p) during training to maintain expected value
    elif not inplace:
        s = s.clone()  # Ensure we return a new tensor when inplace=False

    # If an output tensor is provided, write the result to it
    if out is not None:
        out.copy_(s)

    return s
