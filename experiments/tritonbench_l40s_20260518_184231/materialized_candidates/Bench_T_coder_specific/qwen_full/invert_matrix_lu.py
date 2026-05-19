import torch
import triton
import triton.language as tl

from flag_gems.utils.shape_utils import volume

# This kernel is based on:
# https://github.com/haichuan-li/lu_inverse/blob/master/lu_inverse.cu
@triton.jit
def invert_matrix_lu_kernel(
    # Inputs
    A,  # shape is (batch, n, n)
    pivots,  # shape is (batch, n)
    # Output
    A_inv,  # shape is (batch, n, n)
    # Meta-params
    n: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    # The computation is 2 * n^3 / 6 flops
    pid = tl.program_id(0)
    # each program handles n / num_warps rows
    row_start = tl.program_id(0) * (n // num_warps)
    rows = row_start + tl.arange(0, n // num_warps)
    # The column indices are handled by the 1D block of threads
    cols = tl.arange(0, tl.num_programs(1))
    # Load A(r, c)
    a = tl.load(A + pid * n * n + rows[:, None] * n + cols[None, :])
    # The elements of the row are handled by the 1D block of threads
    row = tl.load(A + pid * n * n + rows[:, None] * n + cols[None, :])
    # Load P(r, c)
    p = tl.load(pivots + pid * n + rows, mask=rows < n, other=0)
    # P is a permutation matrix, so it can be loaded as int
    p = p.to(tl.int32)
    # Compute L(r, c) = I * P(r, c)
    # The first column is handled by the block of threads
    col = tl.load(p + cols, mask=cols < n, other=0)
    col = col.to(tl.int32)
    # A(r, c) = A(r, c) * P(r, c)
    a = tl.where(col[None, :] == rows[:, None], a, 0)
    # Load U(r, c)
    u = tl.load(A + pid * n * n + rows[:, None] * n + cols[None, :])
    # Forward sweep: compute L(r, c)
    for k in range(row_start, row_start + n // num_warps):
        # Multiply A(r, :) by U(k, k)^{-1}
        a = a - tl.dot(row, u.T)
        # Swap A(r, :) and A(k, :)
        a = tl.where(k == rows[:, None], u, tl.where(rows[:, None] == k, a, 0))
        # Load next U(k, :)
        u = tl.load(A + pid * n * n + k * n + cols[None, :])
    # Normalize L(r, c) = L(r, c) * U(k, k)^{-1}
    a = a / u
    # Store L(r, c)
    tl.store(A_inv + pid * n * n + rows[:, None] * n + cols[None, :], a)

def invert_matrix_lu(A, *, pivot=True, out=None):
    """
    Computes the inverse of a square matrix.

    Args:
        A (Tensor): the square matrix, shape must be [..., n, n]
        pivot (bool, optional): whether to use partial pivoting. Default: True
        out (Tensor, optional): the output inverse, if not given, a new tensor is created

    Returns:
        Tensor: the inverse of A
    """
    A = A.contiguous()
    assert A.dim() >= 2
    n = A.size(-1)
    assert A.size(-2) == n
    # Allocates output.
    if out is None:
        A_inv = torch.empty_like(A)
    else:
        A_inv = out
        assert A_inv.shape == A.shape
    # reshaped A and A_inv to be 2D tensors for easy indexing
    A = A.reshape(-1, n, n)
    A_inv = A_inv.reshape(-1, n, n)
    # The kernel is called with grid=(A.size(0),)
    if pivot:
        pivots = torch.empty((A.size(0), n), dtype=torch.int32, device=A.device)
        lu_with_pivoting_kernel[A.size(0)](
            A, pivots, n=n, num_stages=4, num_warps=4
        )
        invert_matrix_lu_kernel[A.size(0)](
            A, pivots, A_inv, n=n, num_stages=4, num_warps=4
        )
    else:
        lu_no_pivoting_kernel[A.size(0)](A, A_inv, n=n, num_stages=4, num_warps=4)
    # flatten the output if the input was 2D
    if A_inv.dim() == 3 and A.dim() == 2:
        A_inv = A_inv.reshape(A_inv.shape[0] * A_inv.shape[1], A_inv.shape[2])
    return A_inv
