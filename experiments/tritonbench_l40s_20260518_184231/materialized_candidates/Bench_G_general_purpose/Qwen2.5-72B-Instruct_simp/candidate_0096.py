import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd_fused(
    x_ptr,  # Pointer to the input tensor
    y_ptr,  # Pointer to the output tensor
    weight_ptr,  # Pointer to the weight vector
    M,  # Number of rows in the input tensor
    N,  # Number of columns in the input tensor
    eps,  # Small epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    row_id = tl.program_id(0)
    if row_id >= M:
        return

    # Pointers to the current row in x and y
    x_row_ptr = x_ptr + row_id * N
    y_row_ptr = y_ptr + row_id * N

    # Load the row of x into a Triton block
    x_row = tl.load(x_row_ptr, mask=tl.arange(0, BLOCK_SIZE) < N, other=0.0)

    # Compute the variance of the row
    mean = tl.sum(x_row, axis=0) / N
    var = tl.sum((x_row - mean) ** 2, axis=0) / N

    # Compute the reciprocal of the square root of the variance plus epsilon
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Load the weight vector
    weight = tl.load(weight_ptr, mask=tl.arange(0, BLOCK_SIZE) < N, other=1.0)

    # Apply the normalization and weight
    y_row = (x_row - mean) * inv_std * weight

    # Store the result back to the output tensor
    tl.store(y_row_ptr, y_row, mask=tl.arange(0, BLOCK_SIZE) < N)

import torch
import triton
import triton.language as tl

# Define the block size for parallelization
BLOCK_SIZE = 1024

def rmsnorm_forward(x, weight, eps=1e-6):
    # Ensure x and weight are on the same device
    assert x.device == weight.device, "x and weight must be on the same device"
    assert x.dim() == 2, "x must be a 2D tensor"
    assert weight.dim() == 1, "weight must be a 1D tensor"
    assert x.size(1) == weight.size(0), "The second dimension of x must match the size of weight"

    M, N = x.size()
    y = torch.empty_like(x, device=x.device)

    # Launch the Triton kernel
    grid = (M,)
    _rms_norm_fwd_fused[grid](
        x, y, weight, M, N, eps, BLOCK_SIZE
    )

    return y
