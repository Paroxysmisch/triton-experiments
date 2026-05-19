import triton
import triton.language as tl

@triton.jit
def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, out=None):
    """
    Performs a fused operation combining element-wise addition, element-wise multiplication,
    and group normalization.

    Args:
        input1: The first input tensor X.
        input2: The second input tensor Y, must be broadcastable to the shape of X.
        weight: Learnable weight parameter γ of shape (C,).
        bias: Learnable bias parameter β of shape (C,).
        num_groups: Number of groups to separate the channels into for group normalization.
        eps: A value added to the denominator for numerical stability in group normalization.
        out: Output tensor. Ignored if None.

    Returns:
        Tensor after performing the fused operation.
    """
    # Get the shape of the input tensors
    shape = input1.shape
    C = shape[1]  # Assuming input1 is of shape [N, C, H, W]

    # Element-wise addition
    Z = input1 + input2  # Broadcasting is handled automatically

    # Element-wise multiplication
    M = Z * input2  # Element-wise multiplication

    # Group normalization
    # Reshape M for group normalization
    M_reshaped = M.reshape(-1, num_groups, C // num_groups, *shape[2:])  # Reshape for group normalization
    mean = tl.mean(M_reshaped, axis=(1, 2), keepdims=True)
    var = tl.var(M_reshaped, axis=(1, 2), keepdims=True)
    inv_std = tl.rsqrt(var + eps)

    # Apply group normalization
    O = weight[None, :, None, None] * (M - mean) * inv_std + bias[None, :, None, None]

    # Write output if out tensor is provided
    if out is not None:
        out.copy_(O)
        return out

    return O
