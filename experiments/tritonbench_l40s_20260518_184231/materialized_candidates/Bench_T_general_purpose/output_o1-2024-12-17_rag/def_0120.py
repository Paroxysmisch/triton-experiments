import triton
import triton.language as tl
import torch

# ------------------------------------------------------------------------------------
# KERNEL 1: Update y in-place with y = alpha * (A @ x) + beta * y
# ------------------------------------------------------------------------------------
@triton.jit
def _matvec_update_kernel(
    A_ptr,   # ptr to A (n, m)
    x_ptr,   # ptr to x (m,)
    y_ptr,   # ptr to y (n,)
    alpha,   # scalar alpha
    beta,    # scalar beta
    n,       # number of rows in A
    m,       # number of cols in A (and size of x)
    BLOCK_SIZE_M: tl.constexpr
):
    # Row index for this program
    row_id = tl.program_id(0)
    # If row is out of range, skip
    if row_id >= n:
        return

    # Accumulate partial dot-product
    # We iterate over columns in blocks of BLOCK_SIZE_M
    col_start = 0
    acc = tl.zeros((1,), dtype=tl.float32)
    while col_start < m:
        offs = col_start + tl.arange(0, BLOCK_SIZE_M)
        # Mask off columns that are out of range
        mask = offs < m

        # Load a block of A[row_id, col_start: col_start+BLOCK_SIZE_M]
        A_block = tl.load(A_ptr + row_id * m + offs, mask=mask, other=0.0)
        # Load a block of x[col_start: col_start+BLOCK_SIZE_M]
        x_block = tl.load(x_ptr + offs, mask=mask, other=0.0)

        # Accumulate
        acc += tl.sum(A_block * x_block, axis=0)

        col_start += BLOCK_SIZE_M

    # Load original y[row_id]
    y_old = tl.load(y_ptr + row_id)
    # Compute new value of y[row_id]
    y_new = alpha * acc[0] + beta * y_old
    # Store update back into y
    tl.store(y_ptr + row_id, y_new)


# ------------------------------------------------------------------------------------
# KERNEL 2: Compute dot(y, x) over range(n). NOTE: n must match length(y) and length(x)
# for a valid dot product, but we assume they match based on the function spec.
# ------------------------------------------------------------------------------------
@triton.jit
def _dot_kernel(
    y_ptr,   # ptr to y (n,)
    x_ptr,   # ptr to x (n,)
    out_ptr, # ptr to 1-element output
    n,       # size of y,x
    BLOCK_SIZE_N: tl.constexpr
):
    # Index for this program
    block_start = tl.program_id(0) * BLOCK_SIZE_N
    offs = block_start + tl.arange(0, BLOCK_SIZE_N)
    mask = offs < n

    # Load and multiply
    y_vals = tl.load(y_ptr + offs, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
    part_sum = tl.sum(y_vals * x_vals, axis=0)

    # Atomically accumulate into out_ptr[0]
    tl.atomic_add(out_ptr, part_sum)


def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Computes y = alpha * torch.mv(A, x) + beta * y (in-place) and returns torch.dot(y, x).
    A is (n, m), x is (m,), y is (n,).
    """
    # Shapes and device checks
    assert A.dim() == 2, "A must be 2D"
    assert x.dim() == 1, "x must be 1D"
    assert y.dim() == 1, "y must be 1D"
    n, m = A.shape
    assert x.shape[0] == m, "x's length must match A's number of columns"
    assert y.shape[0] == n, "y's length must match A's number of rows"
    device = A.device

    # Prepare blocks / grid
    BLOCK_SIZE_M = 128
    # Launch 1D grid over rows of A
    grid_matvec = (n,)

    # Kernel call for y update: y = alpha*(A @ x) + beta*y
    _matvec_update_kernel[grid_matvec](
        A, x, y, 
        alpha, beta, 
        n, m, 
        BLOCK_SIZE_M=BLOCK_SIZE_M
    )

    # Prepare for dot product
    # We'll accumulate into a single-element buffer on device
    out = torch.zeros(1, device=device, dtype=torch.float32)
    BLOCK_SIZE_N = 128
    grid_dot = ( (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N, )

    # Kernel call for dot computation
    _dot_kernel[grid_dot](
        y, x, 
        out, 
        n, 
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )

    # Return the final dot (as a torch scalar Tensor)
    return out[0]
