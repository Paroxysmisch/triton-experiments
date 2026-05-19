import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X,
    Y,
    Rstd,
    stride_ml,
    stride_n,
    L,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Implements a forward kernel for L2 normalization.
    
    Parameters:
    X (tl.tensor): Input tensor where each column represents a feature.
    Y (tl.tensor): Output tensor for normalized features.
    Rstd (tl.tensor): Tensor to store reciprocal of the computed standard deviations.
    stride_ml (int): Stride to access elements along the combined dimensions M and L.
    stride_n (int): Stride to access elements along dimension N.
    L (int): Size of the second dimension in the batch.
    N (int): Total number of features per instance.
    eps (float): Small epsilon value for numerical stability in division.
    BLOCK_SIZE (tl.constexpr): Block size used for partitioning computations.
    """
    # Setup for batched execution over M and L
    row = tl.program_id(0)
    batch = tl.program_id(1)

    # Calculate the base index for the current matrix slice
    base_idx = row * stride_ml + batch * stride_n
    Y += base_idx
    X += base_idx

    _sum_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _sum_squares += a * a
    sum_squares = tl.sum(_sum_squares)
    rstd = 1.0 / tl.sqrt(sum_squares + eps)

    # Store the reciprocal of the standard deviation
    tl.store(Rstd + row * L + batch, rstd)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        tl.store(Y + cols, x_hat, mask=mask)

@triton.jit
def _l2_norm_bwd_kernel(
    input_ptr: tl.pointer_type,
    grad_output_ptr: tl.pointer_type,
    grad_input_ptr: tl.pointer_type,
    rstd_ptr: tl.pointer_type,
    input_row_stride: tl.uint32,
    grad_input_row_stride: tl.uint32,
    num_elements: tl.uint32,
    eps: tl.float32,
    block_size: tl.constexpr,
):
    """
    Implements a backward kernel for L2 normalization.
    
    Parameters:
    input_ptr (tl.pointer_type): Pointer to the input tensor.
    grad_output_ptr (tl.pointer_type): Pointer to the gradient of the output tensor.
    grad_input_ptr (tl.pointer_type): Pointer to the gradient of the input tensor.
    rstd_ptr (tl.pointer_type): Pointer to the reciprocal of the standard deviations.
    input_row_stride (tl.uint32): Stride to access elements along the input row.
    grad_input_row_stride (tl.uint32): Stride to access elements along the gradient input row.
    num_elements (tl.uint32): Number of elements in the row.
    eps (tl.float32): Small epsilon value for numerical stability in division.
    block_size (tl.constexpr): Block size used for partitioning computations.
    """
    # Calculate the row index for this program instance
    row_idx = tl.program_id(0)

    # Create an array of offsets within the block
    offsets = tl.arange(0, block_size)

    # Calculate memory access ranges for the inputs and gradients
    input_offsets = row_idx * input_row_stride + offsets
    grad_output_offsets = row_idx * input_row_stride + offsets
    grad_input_offsets = row_idx * grad_input_row_stride + offsets

    # Create masks to handle cases where block size may exceed the number of elements
    valid_elements_mask = offsets < num_elements

    # Load input values, gradients, and reciprocal standard deviations using the computed offsets and masks
    input_values = tl.load(input_ptr + input_offsets, mask=valid_elements_mask, other=0)
    grad_outputs = tl.load(grad_output_ptr + grad_output_offsets, mask=valid_elements_mask, other=0)
    rstd = tl.load(rstd_ptr + row_idx, mask=valid_elements_mask, other=0)

    # Compute the normalization factor from the input values
    norm_factor = 1.0 / (tl.sqrt(tl.sum(input_values * input_values) / num_elements + eps))

    # Compute partial gradients with respect to input values
    grad_input_first_term = grad_outputs * rstd
    grad_input_second_term = (
        tl.sum(input_values * grad_outputs) * input_values * rstd * rstd * rstd
    )
    grad_input_values = grad_input_first_term - grad_input_second_term
    tl.store(
        grad_input_ptr + grad_input_offsets, grad_input_values, mask=valid_elements_mask
    )

import torch

def _l2_norm_fwd(X, Y, Rstd, stride_ml, stride_n, L, N, eps, BLOCK_SIZE):
    assert X.is_contiguous()
    assert Y.is_contiguous()
    assert Rstd.is_contiguous()
    grid = (X.shape[0], X.shape[1])
    _l2_norm_fwd_1pass_kernel[grid](
        X, Y, Rstd, stride_ml, stride_n, L, N, eps, BLOCK_SIZE
    )

def _l2_norm_bwd(X, DY, DX, Rstd, stride_ml, stride_n, L, N, eps, BLOCK_SIZE):
    assert X.is_contiguous()
    assert DY.is_contiguous()
    assert DX.is_contiguous()
    assert Rstd.is_contiguous()
    grid = (X.shape[0],)
    _l2_norm_bwd_kernel[grid](
        X, DY, DX, Rstd, stride_ml, stride_n, N, eps, BLOCK_SIZE
    )

import torch

# Example input tensor
X = torch.randn(32, 64, device='cuda')

# Output tensor and reciprocal standard deviations
Y = torch.empty_like(X)
Rstd = torch.empty(X.shape[0] * X.shape[1], device='cuda')

# Forward pass
_l2_norm_fwd(X, Y, Rstd, X.stride(0), X.stride(1), X.shape[1], X.shape[2], 1e-6, 128)

# Example gradient tensor
DY = torch.randn_like(Y)

# Gradient output tensor
DX = torch.empty_like(X)

# Backward pass
_l2_norm_bwd(X, DY, DX, Rstd, X.stride(0), X.stride(1), X.shape[1], X.shape[2], 1e-6, 128)
