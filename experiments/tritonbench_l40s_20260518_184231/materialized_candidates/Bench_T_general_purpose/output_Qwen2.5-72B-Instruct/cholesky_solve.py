import triton
import triton.language as tl

@triton.jit
def cholesky_solve_kernel(
    B_ptr,  # Pointer to the right-hand side tensor
    L_ptr,  # Pointer to the Cholesky decomposition tensor
    X_ptr,  # Pointer to the output tensor
    B_batch_stride, B_m_stride, B_n_stride,
    L_batch_stride, L_m_stride, L_n_stride,
    X_batch_stride, X_m_stride, X_n_stride,
    n, k,  # Dimensions
    upper,  # Flag indicating whether L is upper triangular
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch = pid // (n * k)
    b = (pid % (n * k)) // k
    j = (pid % (n * k)) % k

    # Pointers for the current batch
    B_batch_ptr = B_ptr + batch * B_batch_stride
    L_batch_ptr = L_ptr + batch * L_batch_stride
    X_batch_ptr = X_ptr + batch * X_batch_stride

    # Initialize the output vector
    x = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for i in range(n):
        # Load the current row of L
        L_row = tl.load(L_batch_ptr + i * L_m_stride + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n, other=0.0)

        if upper:
            # Upper triangular case
            if i > b:
                break
            x[i] = (tl.load(B_batch_ptr + b * B_m_stride + j * B_n_stride) - tl.dot(L_row[b:], x[b:])) / L_row[b]
        else:
            # Lower triangular case
            if i < b:
                continue
            x[i] = (tl.load(B_batch_ptr + b * B_m_stride + j * B_n_stride) - tl.dot(L_row[:i], x[:i])) / L_row[i]

    # Store the result
    tl.store(X_batch_ptr + b * X_m_stride + j * X_n_stride, x[b], mask=b < n)

import torch
import triton
import triton.language as tl

def cholesky_solve(B, L, upper=False, *, out=None):
    # Ensure the inputs are on the same device
    device = B.device
    assert L.device == device, "B and L must be on the same device"

    # Get the dimensions
    *batch, n, k = B.shape
    *batch, n, n = L.shape

    # Ensure the batch dimensions match
    assert B.shape[:-2] == L.shape[:-2], "B and L must have the same batch dimensions"

    # Determine the data type
    dtype = B.dtype
    assert dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128], "Unsupported data type"

    # Allocate the output tensor if not provided
    if out is None:
        out = torch.empty_like(B, device=device, dtype=dtype)
    else:
        assert out.shape == B.shape, "Output tensor must have the same shape as B"
        assert out.dtype == dtype, "Output tensor must have the same data type as B"

    # Launch the Triton kernel
    grid = (n * k * (1 if len(batch) == 0 else batch[0]),)
    cholesky_solve_kernel[grid](
        B, L, out,
        B.stride(0) if len(batch) > 0 else 0, B.stride(-2), B.stride(-1),
        L.stride(0) if len(batch) > 0 else 0, L.stride(-2), L.stride(-1),
        out.stride(0) if len(batch) > 0 else 0, out.stride(-2), out.stride(-1),
        n, k, upper, BLOCK_SIZE=128
    )

    return out
