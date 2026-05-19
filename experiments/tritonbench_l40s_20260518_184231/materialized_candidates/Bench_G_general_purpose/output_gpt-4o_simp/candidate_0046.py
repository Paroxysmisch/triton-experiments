import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(X_ptr, Y_ptr, stride_x, stride_y, N, BLOCK_SIZE: tl.constexpr):
    # Pointers to input and output rows
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x
    Y_row_ptr = Y_ptr + row_idx * stride_y

    # Load data from the row
    offsets = tl.arange(0, BLOCK_SIZE)
    X_row = tl.load(X_row_ptr + offsets, mask=offsets < N, other=0.0)

    # Compute the squared norm of the row
    X_squared = X_row * X_row
    sum_X_squared = tl.sum(X_squared, axis=0)

    # Compute L2 norm
    norm = tl.sqrt(sum_X_squared)

    # Normalize the row
    Y_row = X_row / norm

    # Store the result
    tl.store(Y_row_ptr + offsets, Y_row, mask=offsets < N)

@triton.jit
def _l2_norm_bwd_kernel(X_ptr, DY_ptr, DX_ptr, stride_x, stride_dy, stride_dx, N, BLOCK_SIZE: tl.constexpr):
    # Pointers to input, gradient, and output rows
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x
    DY_row_ptr = DY_ptr + row_idx * stride_dy
    DX_row_ptr = DX_ptr + row_idx * stride_dx

    # Load data from the row
    offsets = tl.arange(0, BLOCK_SIZE)
    X_row = tl.load(X_row_ptr + offsets, mask=offsets < N, other=0.0)
    DY_row = tl.load(DY_row_ptr + offsets, mask=offsets < N, other=0.0)

    # Compute the squared norm of the row
    X_squared = X_row * X_row
    sum_X_squared = tl.sum(X_squared, axis=0)

    # Compute L2 norm
    norm = tl.sqrt(sum_X_squared)

    # Compute gradient w.r.t input
    X_norm = X_row / norm
    sum_XDY = tl.sum(X_norm * DY_row, axis=0)
    DX_row = (DY_row - X_norm * sum_XDY) / norm

    # Store the result
    tl.store(DX_row_ptr + offsets, DX_row, mask=offsets < N)

def _l2_norm_fwd(X):
    # Ensure X is a 2D tensor
    assert X.ndim == 2
    M, N = X.shape
    Y = torch.empty_like(X)

    # Launch Triton kernel
    BLOCK_SIZE = 1024  # Adjust block size as needed
    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](
        X, Y,
        X.stride(0), Y.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return Y

def _l2_norm_bwd(X, DY):
    # Ensure X and DY are 2D tensors
    assert X.ndim == 2 and DY.ndim == 2
    M, N = X.shape
    DX = torch.empty_like(X)

    # Launch Triton kernel
    BLOCK_SIZE = 1024  # Adjust block size as needed
    grid = (M,)
    _l2_norm_bwd_kernel[grid](
        X, DY, DX,
        X.stride(0), DY.stride(0), DX.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return DX
