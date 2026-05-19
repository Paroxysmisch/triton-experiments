import torch
import triton
import triton.language as tl

@triton.jit
def _fused_cholesky_solve_kernel(A, b, n, k, stride_am, stride_ak, stride_bk, stride_bn):
    row, col = tl.program_id(0), tl.program_id(1)
    if row < n:
        # Compute L * y = b
        sum_ = 0.0
        for m in range(0, col):
            l = tl.load(A + row * stride_am + m * stride_ak)
            y = tl.load(b + m * stride_bk + col * stride_bn)
            sum_ += l * y
        y = tl.load(b + col * stride_bk + row * stride_bn)
        y -= sum_
        y /= tl.load(A + row * stride_am + row * stride_ak)
        tl.store(b + col * stride_bk + row * stride_bn, y)

        # Compute L.T * x = y
        for m in range(col + 1, n):
            l = tl.load(A + m * stride_am + row * stride_ak)
            y = tl.load(b + col * stride_bk + m * stride_bn)
            y -= l * tl.load(b + col * stride_bk + row * stride_bn)
            tl.store(b + col * stride_bk + m * stride_bn, y)
        x = tl.load(b + col * stride_bk + row * stride_bn)
        for m in range(row + 1, n):
            x -= tl.load(A + m * stride_am + row * stride_ak) * tl.load(b + col * stride_bk + m * stride_bn)
        x /= tl.load(A + row * stride_am + row * stride_ak)
        tl.store(b + col * stride_bk + row * stride_bn, x)
    return

def fused_cholesky_solve(A: Tensor, b: Tensor) -> Tensor:
    n, k = b.shape
    assert A.shape[0] == A.shape[1] == n, "Matrix A must be square"
    assert A.stride(0) == 1 or A.stride(1) == 1, "Matrix A must be contiguous in memory"
    assert b.stride(0) == 1 or b.stride(1) == 1, "Matrix b must be contiguous in memory"
    A = A.contiguous()
    b = b.contiguous()
    _fused_cholesky_solve_kernel[(n, n)](A, b, n, k, A.stride(0), A.stride(1), b.stride(0), b.stride(1))
    return b
