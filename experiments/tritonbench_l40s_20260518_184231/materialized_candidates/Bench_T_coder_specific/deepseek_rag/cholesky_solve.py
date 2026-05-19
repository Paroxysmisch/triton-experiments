import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_solve_kernel(B, L, Y, M, N, upper=False):
    pid = tl.program_id(0)
    row = pid * M + tl.arange(0, M)
    mask = row < M
    B += row * N
    L += row * M
    Y += row * N

    if upper:
        L = L.T

    X = tl.zeros((M,), dtype=B.dtype)
    for n in range(N):
        b = tl.load(B + n, mask, other=0.0)
        for m in range(M):
            L_mm = L[m, m]
            x_m = X[m]
            x_m = (b[m] - tl.dot(L[m, :m], X[:m])) / L_mm
            X[m] = x_m
        Y[n] = X

def cholesky_solve(B, L, upper=False, *, out=None):
    assert len(B.shape) == 3, "Right-hand side tensor must have 3 dimensions"
    M, _, N = B.shape
    if out is None:
        out = torch.empty_like(B)
    else:
        assert out.shape == B.shape, "Output tensor must have the same shape as the right-hand side tensor"
    with torch.cuda.device(B.device):
        grid = lambda meta: (triton.cdiv(M * N, meta["M"]),)
        cholesky_solve_kernel[grid](B, L, out, M, N, upper)
    return out
