import triton
import triton.language as tl
import math

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none', *, out=None):
    """
    Perform a fused operation combining batch matrix multiplication, RMS normalization, GELU activation, and dropout.
    
    Args:
        input1 (Tensor): First input tensor for bmm, shape (B, N, M)
        input2 (Tensor): Second input tensor for bmm, shape (B, M, P)
        normalized_shape (int or list or torch.Size): Shape over which RMS normalization is applied
        dropout_p (float, optional): Probability of dropout. Default: 0.1
        eps (float, optional): Value for numerical stability in RMSNorm. Default: 1e-5
        training (bool, optional): Whether to apply dropout. Default: True
        approximate (str, optional): 'none' or 'tanh' for GELU approximation. Default: 'none'
        out (Tensor, optional): Output tensor. Default: None
    """
    
    # Step 1: Perform BMM (batch matrix multiplication)
    Z1 = tl.dot(input1, input2)  # Result shape will be (B, N, P)

    # Step 2: RMS Normalization
    # Calculate squared values for RMS
    squared_Z1 = tl.square(Z1)
    rms_norm = tl.sqrt(tl.mean(squared_Z1, axis=-1) + eps)
    Z2 = Z1 / rms_norm[:, :, None]  # Normalize over the last dimension (P)

    # Step 3: Apply GELU activation
    if approximate == 'tanh':
        Z3 = 0.5 * Z2 * (1.0 + tl.tanh(math.sqrt(2 / math.pi) * (Z2 + 0.044715 * Z2 ** 3)))
    else:
        Z3 = Z2 * 0.5 * (1.0 + tl.erf(Z2 / math.sqrt(2)))  # GELU activation
    
    # Step 4: Apply Dropout
    if training:
        mask = tl.random.uniform(shape=Z3.shape, dtype=tl.float32) > dropout_p
        Z3 = Z3 * mask  # Dropout with probability p
    
    # Output result
    if out is not None:
        out[:] = Z3  # Store the result into 'out' if provided
    return Z3
