import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def fused_layer_norm_relu_linear_kernel(
    input,
    weight,
    bias,
    output,
    normalized_shape,
    eps,
    elementwise_affine,
    n_cols,
    BLOCK_N: tl.constexpr,
):
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    # Load input
    x = tl.load(input + prog_id * n_cols + offsets)
    xf = x.to(tl.float32)

    # Linear transformation
    w = tl.load(weight + offsets)
    y = tl.dot(w, xf)
    
    if bias is not None:
        b = tl.load(bias)
        y += b

    # ReLU activation
    y = tl.maximum(y, 0)

    # Layer normalization
    mean = tl.sum(y) / n_cols
    var = tl.sum((y - mean) ** 2) / n_cols
    y = (y - mean) / tl.sqrt(var + eps)

    if elementwise_affine:
        y *= tl.load(weight + offsets)  # Scale with weight

    tl.store(output + prog_id * n_cols + offsets, y)

@torch.inference_mode()
def fused_layer_norm_relu_linear(input: Tensor, weight: Tensor, bias: Tensor = None, normalized_shape: int = None, eps: float = 1e-5, elementwise_affine: bool = True) -> Tensor:
    """
    Applies a fused operation consisting of a linear transformation followed by ReLU activation and layer normalization on the input tensor.

    Args:
        input (Tensor): Input tensor with shape (*, in_features).
        weight (Tensor): Weights for the linear transformation, shape (out_features, in_features).
        bias (Tensor, optional): Bias for the linear transformation, shape (out_features).
        normalized_shape (int or list or torch.Size, optional): Shape of the dimensions to normalize.
        eps (float, optional): A value added to the denominator for numerical stability. Default is 1e-5.
        elementwise_affine (bool, optional): If True, layer normalization has learnable parameters. Default is True.

    Returns:
        Tensor: Result after applying the linear transformation, ReLU, and layer normalization.
    """
    seq_len = input.size(0)
    n_cols = weight.size(0)
    BLOCK_N = triton.next_power_of_2(n_cols)
    output = torch.empty((seq_len, n_cols), device=input.device, dtype=input.dtype)

    grid = (seq_len,)
    fused_layer_norm_relu_linear_kernel[grid](
        input,
        weight,
        bias,
        output,
        normalized_shape,
        eps,
        elementwise_affine,
        n_cols,
        BLOCK_N,
        num_warps=4,
        num_stages=2,
        device=input.device,
    )
    return output
