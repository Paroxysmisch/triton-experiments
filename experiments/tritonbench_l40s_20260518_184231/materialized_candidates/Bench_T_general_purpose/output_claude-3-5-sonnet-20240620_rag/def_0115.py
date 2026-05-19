import torch
import triton
import triton.language as tl

@triton.jit
def kernel_tanh_fwd(
    C,  # Output tensor
    A,  # Input tensor
    B,  # Weight matrix
    bias,  # Optional bias
    # Matrix dimensions
    M,  # Number of input rows
    N,  # Number of output columns
    K,  # Number of input columns (features)
    stride_cm,
    stride_am,
    stride_bk,
    stride_bn,
):
    """
    Kernel for computing Out = tanh(A @ B^T + bias)
    - Input has shape (M, K)
    - Weight has shape (N, K)
    - Bias has shape (N,)
    - Output has shape (M, N)
    """
    pid = tl.program_id(axis=0)

    # Calculate grid dimensions
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # Compute the block that each program will go through
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)

    # Load input and weight
    a = tl.load(A + rm[:, None] * stride_am)
    b = tl.load(B + rn[None, :] * stride_bn)

    # Compute linear transformation
    acc = tl.dot(a, b)

    # Add bias if provided
    if bias is not None:
        bias_val = tl.load(bias + rn)
        acc += bias_val[None, :]

    # Apply Tanh activation
    out = (tl.exp(acc) - tl.exp(-acc)) / (tl.exp(acc) + tl.exp(-acc))

    # Store the result
    tl.store(C + rm[:, None] * stride_cm, out)

def tanh_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation followed by a Tanh activation function.
    :param input: Input tensor of shape (*, in_features)
    :param weight: Weight matrix of shape (out_features, in_features)
    :param bias: Optional bias tensor of shape (out_features)
    :return: Output tensor after applying linear transformation and Tanh activation
    """
    batch_shape, n = input.shape[:-1], input.shape[-1]
    batch_dim = batch_shape.numel()
    input_reshaped = input.reshape(batch_dim, n)

    M, K = input_reshaped.shape
    N = weight.shape[0]

    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    # Launch kernel
    kernel_tanh_fwd[(M, N)](
        output,
        input_reshaped,
        weight,
        bias,
        M,
        N,
        K,
        output.stride(0),
        input_reshaped.stride(0),
        weight.stride(1),
        weight.stride(0),
    )

    return output.reshape(*batch_shape, output.shape[-1])
