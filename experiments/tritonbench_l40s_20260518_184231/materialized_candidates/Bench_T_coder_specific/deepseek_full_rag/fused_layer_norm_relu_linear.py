import torch
import triton
import triton.language as tl

@triton.jit
def fused_layer_norm_relu_linear_kernel(
    input,
    weight,
    bias,
    out,
    input_row_stride,
    n_cols,
    eps,
    elementwise_affine,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    # compute mean and variance
    x_ptr = input + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < n_cols)
    xf = x.to(tl.float32)
    mean = tl.sum(xf, 0) * float(1.0 / N_COLS)
    xcentered = xf - mean
    var = tl.sum(xcentered * xcentered, 0) * float(1.0 / N_COLS)
    rstd = tl.math.rsqrt(var + eps)

    # normalize, optionally affine, and ReLU
    x_norm = xcentered * rstd
    w = tl.load(weight + offsets, mask=offsets < n_cols)
    if elementwise_affine:
        b = tl.load(bias + offsets, mask=offsets < n_cols)
        x_norm = x_norm * w + b
    else:
        x_norm = x_norm * w
    out = tl.where(x_norm > 0, x_norm, 0.0)

    # write output
    out_ptr = out + prog_id * input_row_stride
    tl.store(out_ptr + offsets, out, mask=offsets < n_cols)


def fused_layer_norm_relu_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    normalized_shape: Optional[Union[int, List[int], torch.Size]] = None,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
) -> torch.Tensor:
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

    Example:
        >>> input = torch.randn(4, 5) # Example input tensor
        >>> weight = torch.randn(3, 5) # Linear transformation weights
        >>> bias = torch.randn(3) # Bias for linear layer
        >>> normalized_shape = 3
        >>> # Apply fused operation
        >>> output = fused_layer_norm_relu_linear(input, weight, bias, normalized_shape)
        >>> print(output.shape) # Expected output shape: (4, 3)
    """

    def _kernel_meta():
        device = input.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    if input.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")

    if normalized_shape is None:
        normalized_shape = input.shape[-1]

    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    else:
        normalized_shape = tuple(normalized_shape)

    n_cols = input.shape[-1]

    for dim in normalized_shape:
        if dim != n_cols:
            raise ValueError(
                "Invalid normalized_shape: {} for input of shape {}".format(
                    normalized_shape, input.shape
                )
            )

    if elementwise_affine:
        if bias is None:
            bias = torch.empty_like(weight)
        if (
            bias.shape[0] != weight.shape[0]
            or bias.shape[1] != weight.shape[1]
            or weight.ndim != 2
            or bias.ndim != 2
        ):
            raise ValueError(
                "Incompatible dimensions between weight, bias, and normalized shape"
            )

    input_stride = input.stride(-2)
    out = torch.empty_like(input)

    BLOCK_N = triton.next_power_of_2(n_cols)
    grid = (input.numel() // input_stride,)
    kernel_meta = _kernel_meta()
    fused_layer_norm_relu_linear_kernel[grid](
        input,
        weight,
        bias,
        out,
        input_stride,
        n_cols,
        eps,
        elementwise_affine,
        n_cols,
        BLOCK_N,
        num_warps=4,
        num_stages=2,
        **kernel_meta,
    )
    return out
