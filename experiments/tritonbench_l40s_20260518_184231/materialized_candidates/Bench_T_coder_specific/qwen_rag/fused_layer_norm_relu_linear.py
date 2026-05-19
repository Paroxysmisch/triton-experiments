import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def fused_layer_norm_relu_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_row_stride,
    n_in_features,
    n_out_features,
    normalized_shape,
    eps,
    elementwise_affine,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // BLOCK_SIZE
    col = pid % BLOCK_SIZE

    if row >= normalized_shape or col >= n_in_features:
        return

    x = tl.load(input_ptr + row * input_row_stride + col)
    w = tl.load(weight_ptr + col)
    b = bias_ptr[col] if elementwise_affine else 0.0

    y = w * x + b
    y = max(y, 0.0)  # ReLU activation

    mean = tl.sum(y, axis=0) / normalized_shape
    variance = tl.sum((y - mean) * (y - mean), axis=0) / normalized_shape + eps
    norm_y = (y - mean) / tl.sqrt(variance)

    output_ptr[row * input_row_stride + col] = norm_y
