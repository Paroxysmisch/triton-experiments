import triton
import triton.language as tl

@triton.jit
def fused_cholesky_solve(A: tl.tensor, b: tl.tensor) -> tl.tensor:
    # Cholesky decomposition
    L = tl.linalg.cholesky(A)
    # Solve Lx = b for x
    return tl.linalg.solve_triangular(L, b, lower=True)
