import torch
import triton
import triton.language as tl
from triton.language.extra import householder

@triton.jit
def fused_qr_solve_kernel(
    A_ptr, b_ptr, x_ptr, m, n, k, nr: tl.constexpr, nc: tl.constexpr
):
    """
    Solves the linear system Ax = b using QR decomposition.
    Args:
        A_ptr: Pointer to the matrix A of shape (m, n).
        b_ptr: Pointer to the right-hand side tensor b of shape (m, k).
        x_ptr: Pointer to the solution tensor x of shape (n, k).
        m: Number of rows in A.
        n: Number of columns in A.
        k: Number of columns in b.
        nr: Number of rows in each tile.
        nc: Number of columns in each tile.
    """
    tl.static_assert(nr % 2 == 0, "nr must be even")
    # Compute the block indices
    i_row = tl.program_id(0)
    j_col = tl.program_id(1)

    i_offs = i_row * nr + tl.arange(0, nr)
    j_offs = j_col * nc + tl.arange(0, nc)

    A = tl.load(A_ptr + i_offs[:, None] * n + j_offs[None, :])
    b = tl.load(b_ptr + i_offs[:, None] * k + j_offs[None, :])

    # QR decomposition
    q, r = householder.qr(A)

    # Solve
    x = tl.dot(tl.inverse(r), tl.trans(q) @ b)

    # Store the result
    tl.store(x_ptr + i_offs[:, None] * k + j_offs[None, :], x)

def fused_qr_solve(A: Tensor, b: Tensor) -> Tensor:
    m, n = A.shape
    n_b, k = b.shape
    assert m >= n
    assert n_b == m

    x = torch.empty((n, k), dtype=A.dtype, device=A.device)

    def grid(meta):
        return (triton.cdiv(m, meta["nr"]), triton.cdiv(n, meta["nc"]))

    fused_qr_solve_kernel[grid](A, b, x, m, n, k)
