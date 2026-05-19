import torch
import triton
import triton.language as tl

@triton.jit
def _add_scaled_vector_kernel(
    x_ptr, y_ptr, alpha, n, k,
    stride_xr, stride_xc, stride_yr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create masks to guard memory loads/stores
    mask_m = row_offsets < n
    mask_n = col_offsets < k

    # Form a 2D meshgrid for row and col
    row = row_offsets[:, None]
    col = col_offsets[None, :]

    # Load x
    x_addr = x_ptr + row * stride_xr + col * stride_xc
    x_vals = tl.load(x_addr, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    # Load y (broadcast across columns)
    y_addr = y_ptr + row * stride_yr
    y_vals = tl.load(y_addr, mask=mask_m[:, None], other=0.0)

    # Add alpha * y to x
    x_new = x_vals + alpha * y_vals

    # Store result back to x
    tl.store(x_addr, x_new, mask=mask_m[:, None] & mask_n[None, :])

def _add_scaled_vector(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    # Ensure 2D shapes for x
    if x.ndim == 1:
        x = x.unsqueeze(1)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    # Shapes
    n, k = x.shape
    # Strides
    stride_xr = x.stride(0)
    stride_xc = x.stride(1)
    stride_yr = y.stride(0)
    # Grid
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (
        ( (n + BLOCK_M - 1) // BLOCK_M ),
        ( (k + BLOCK_N - 1) // BLOCK_N )
    )
    # Launch kernel
    triton.run(
        _add_scaled_vector_kernel,
        grid=grid,
        args=[
            x, y, alpha, n, k,
            stride_xr, stride_xc, stride_yr
        ],
        num_warps=4,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )
    return x

def solve_and_add_scaled_vector(
    A: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    # Solve triangular system
    x = torch.linalg.solve_triangular(A, b, upper=True)
    # Add scaled vector using Triton
    x = _add_scaled_vector(x, y, alpha)
    # Return the final result
    return x
