import triton
import triton.language as tl
import torch

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p, eps, training, approximate, out):
    """
    Performs a fused operation combining batch matrix multiplication,
    RMS normalization, GELU activation, and dropout.

    Args:
        input1: First input tensor for bmm, of shape (B, N, M).
        input2: Second input tensor for bmm, of shape (B, M, P).
        normalized_shape: Shape over which RMS normalization is applied.
        dropout_p: Probability of an element to be zeroed in the dropout layer.
        eps: Value added for numerical stability in RMS normalization.
        training: Apply dropout if True.
        approximate: Approximation to use for GELU ('none' or 'tanh').
        out: Output tensor. Ignored if None.

    Returns:
        Output tensor after applying the fused operations.
    """
    # Step 1: Batch Matrix Multiplication
    Z1 = tl.dot(input1, input2)  # Shape (B, N, P)

    # Step 2: RMS Normalization
    mean = tl.mean(Z1, axis=-1, keepdims=True)
    Z1_centered = Z1 - mean
    rms = tl.sqrt(tl.mean(Z1_centered ** 2, axis=-1, keepdims=True) + eps)
    Z2 = Z1_centered / rms  # RMSNorm output

    # Step 3: GELU Activation
    if approximate == 'tanh':
        Z3 = 0.5 * Z2 * (1 + tl.tanh(tl.sqrt(2 / tl.pi) * (Z2 + 0.044715 * Z2 ** 3)))
    else:
        Z3 = Z2 * 0.5 * (1 + tl.erf(Z2 / tl.sqrt(2)))

    # Step 4: Dropout
    if training:
        mask = tl.rand(Z3.shape) > dropout_p
        Z3 *= mask

    # Step 5: Output
    if out is not None:
        out.copy_(Z3)
    return Z3
