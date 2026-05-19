import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def _fused_layer_norm_relu_linear_kernel(
    x_ptr,       # input pointer
    w_ptr,       # weight pointer (out_features, in_features)
    b_ptr,       # bias pointer (optional)
    o_ptr,       # output pointer
    stride_xm,   # input stride for the batch dimension
    stride_xk,   # input stride for the in_features dimension
    stride_wo,   # weight stride for out_features dimension
    stride_wi,   # weight stride for in_features dimension
    stride_om,   # output stride for the batch dimension
    stride_on,   # output stride for the out_features dimension
    B,           # total batch (flattened) size
    N,           # out_features
    K,           # in_features
    eps,         # layer norm epsilon
    USE_BIAS: tl.constexpr,    # indicates if bias is provided
    ELEM_AFFINE: tl.constexpr, # indicates if layer norm is elementwise affine
):
    # program_id corresponds to batch index
    b_idx = tl.program_id(0)
    # if this index is out of range, we do nothing
    if b_idx >= B:
        return

    # We'll compute the linear transformation in a single pass:
    # out_val[n] = sum_{k=0..K-1} x[b_idx, k] * w[n, k] + (bias[n] if provided)

    # Create an index array for the out_features dimension
    n_offsets = tl.arange(0, N)
    # Initialize accumulator
    acc = tl.zeros([N], dtype=tl.float32)

    # Base ptr for input row and output row
    x_row_ptr = x_ptr + b_idx * stride_xm
    o_row_ptr = o_ptr + b_idx * stride_om

    # Compute the linear transformation with a simple loop over K
    # (This is a naive gemv approach)
    for k in range(K):
        x_val = tl.load(x_row_ptr + k * stride_xk)
        # Load weight row = w[n, k]
        w_col = tl.load(w_ptr + n_offsets * stride_wo + k * stride_wi, mask=n_offsets < N)
        acc += x_val.to(tl.float32) * w_col

    if USE_BIAS:
        b_val = tl.load(b_ptr + n_offsets, mask=n_offsets < N)
        acc += b_val

    # Apply ReLU
    acc = tl.maximum(acc, 0.0)

    # Now apply Layer Normalization across dimension N:
    # mean = sum(acc) / N
    # var  = sum((acc - mean)^2) / N
    # acc  = (acc - mean) / sqrt(var + eps)
    # For simplicity, we do not apply gamma/beta (elementwise_affine) because no gamma/beta provided;
    # if ELEM_AFFINE is True, it would require parameters. Here we assume 1 and 0 for LN scale/shift.
    mean = tl.sum(acc, axis=0) / N
    var = tl.sum((acc - mean) * (acc - mean), axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + eps)
    acc = (acc - mean) * inv_std

    # Store results
    # Each thread covers N elements for this batch index
    tl.store(o_row_ptr + n_offsets * stride_on, acc, mask=n_offsets < N)


@torch.no_grad()
def fused_layer_norm_relu_linear(
    input: Tensor,
    weight: Tensor,
    bias=None,
    normalized_shape=None,
    eps: float = 1e-5,
    elementwise_affine: bool = True
) -> Tensor:
    """
    Applies a fused operation of linear transformation, ReLU, and layer normalization
    to the input tensor.

    Args:
        input (Tensor): Input tensor of shape (*, in_features).
        weight (Tensor): Weight for the linear transformation of shape (out_features, in_features).
        bias (Tensor, optional): Bias for the linear transformation of shape (out_features).
        normalized_shape (int or list or torch.Size, optional): Dimensions to normalize over.
        eps (float, optional): A value added to the denominator for numerical stability. Default: 1e-5.
        elementwise_affine (bool, optional): If True, layer normalization has learnable parameters.
                                             (Not used here; gamma/beta not provided.) Default: True.

    Returns:
        Tensor: Result after applying linear transformation, ReLU, and layer normalization.
    """
    if normalized_shape is None:
        normalized_shape = weight.shape[0]
    # Flatten all dims except the last (in_features) for simplicity
    *leading_dims, k_size = input.shape
    if k_size != weight.shape[1]:
        raise ValueError("Input in_features must match weight.shape[1].")

    batch_size = 1
    for d in leading_dims:
        batch_size *= d
    out_features, in_features = weight.shape

    # Create output tensor of shape [*leading_dims, out_features]
    out_shape = leading_dims + [out_features]
    out = torch.empty(*out_shape, device=input.device, dtype=input.dtype)

    # Strides needed for Triton
    # input strides
    stride_xm = input.stride(0) if len(input.stride()) == 1 else input.stride(-2)  # flatten
    stride_xk = input.stride(-1)
    # weight strides
    stride_wo = weight.stride(0)
    stride_wi = weight.stride(1)
    # output strides
    stride_om = out.stride(0) if len(out.stride()) == 1 else out.stride(-2)
    stride_on = out.stride(-1)

    def _kernel_meta():
        device = input.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    kernel_meta = _kernel_meta()

    grid = (batch_size,)
    _fused_layer_norm_relu_linear_kernel[grid](
        input, weight,
        bias if bias is not None else
