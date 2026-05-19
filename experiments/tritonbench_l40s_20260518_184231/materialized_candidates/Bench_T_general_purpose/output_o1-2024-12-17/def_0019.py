import torch
import triton
import triton.language as tl

@triton.jit
def _naive_lu_decompose_kernel(
    A_ptr, stride, N,
    BLOCKSIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Each block handles one "k" (pivot row)
    # We assume no pivoting for simplicity.
    # The factorization is done in-place.
    if pid < N - 1:
        k = pid
        # Get pivot
        pivot = tl.load(A_ptr + k * stride + k)
        # Divide elements below pivot
        row_idx = k + 1 + tl.arange(0, BLOCKSIZE)
        mask = row_idx < N
        pivot_vals = tl.load(A_ptr + row_idx * stride + k, mask=mask)
        pivot_vals /= pivot
        tl.store(A_ptr + row_idx * stride + k, pivot_vals, mask=mask)

        # Update submatrix
        for j in range(k + 1, N):
            col_vals = tl.load(A_ptr + row_idx * stride + j, mask=mask)
            sub_vals = tl.load(A_ptr + row_idx * stride + k, mask=mask) * tl.load(A_ptr + k * stride + j)
            col_vals -= sub_vals
            tl.store(A_ptr + row_idx * stride + j, col_vals, mask=mask)

@triton.jit
def _forward_substitution_kernel(
    A_ptr, b_ptr, y_ptr, strideA, N,
    BLOCKSIZE: tl.constexpr
):
    # Each block handles one row "i"
    pid = tl.program_id(0)
    if pid < N:
        i = pid
        sum_ = tl.load(b_ptr + i)
        for j in range(i):
            valA = tl.load(A_ptr + i * strideA + j)
            valY = tl.load(y_ptr + j)
            sum_ -= valA * valY
        # L has implicit 1 on diagonal
        tl.store(y_ptr + i, sum_)

@triton.jit
def _backward_substitution_kernel(
    A_ptr, y_ptr, x_ptr, strideA, N,
    BLOCKSIZE: tl.constexpr
):
    # Each block handles one row "i" backward
    pid = tl.program_id(0)
    i = N - pid - 1
    if i >= 0 and i < N:
        sum_ = tl.load(y_ptr + i)
        for j in range(i + 1, N):
            valA = tl.load(A_ptr + i * strideA + j)
            valX = tl.load(x_ptr + j)
            sum_ -= valA * valX
        diag = tl.load(A_ptr + i * strideA + i)
        tl.store(x_ptr + i, sum_ / diag)

def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solves Ax = b using a naive LU factorization (without pivoting) and forward/backward substitution.
    A is decomposed in-place. Returns the solution vector x.
    """
    # Ensure A and b are on CUDA
    if not A.is_cuda:
        A = A.cuda()
    if not b.is_cuda:
        b = b.cuda()

    N = A.shape[0]
    # We will do an in-place factorization, so clone A if needed
    A_work = A.clone()
    strideA = A_work.stride(0)
    
    # Launch one kernel per pivot for naive decomposition
    for k in range(N - 1):
        grid = (1,)
        _naive_lu_decompose_kernel[grid](
            A_work.data_ptr(), strideA, N,
            BLOCKSIZE=1024,
            num_warps=1
        )
    
    # Allocate space for intermediate results y and final x
    y = torch.zeros_like(b)
    x = torch.zeros_like(b)
    
    # Forward substitution
    grid = (N,)
    _forward_substitution_kernel[grid](
        A_work.data_ptr(), b.data_ptr(), y.data_ptr(), strideA, N,
        BLOCKSIZE=1,
        num_warps=1
    )
    
    # Backward substitution
    grid = (N,)
    _backward_substitution_kernel[grid](
        A_work.data_ptr(), y.data_ptr(), x.data_ptr(), strideA, N,
        BLOCKSIZE=1,
        num_warps=1
    )
    
    return x
