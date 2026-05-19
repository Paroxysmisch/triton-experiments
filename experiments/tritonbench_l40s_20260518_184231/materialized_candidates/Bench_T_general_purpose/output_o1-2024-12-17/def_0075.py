import math
import triton
import triton.language as tl

@triton.jit
def _forward_substitution_kernel(
    L_ptr, b_ptr, x_ptr,
    n, k,
    strideL, strideb, stridex,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    for col in range(k):
        # Forward substitution
        for i in range(n):
            # Load b[i, col]
            rhs = tl.load(b_ptr + i * strideb + col)
            # Dot product L[i, 0..i-1] * x[0..i-1, col]
            sum_val = 0.0
            for j in range(i):
                l_val = tl.load(L_ptr + i * strideL + j)
                x_val = tl.load(x_ptr + j * stridex + col)
                sum_val += l_val * x_val
            # x[i, col] = (b[i, col] - sum_val) / L[i, i]
            diag = tl.load(L_ptr + i * strideL + i)
            x_new = (rhs - sum_val) / diag
            tl.store(x_ptr + i * stridex + col, x_new)

@triton.jit
def _backward_substitution_kernel(
    L_ptr, x_ptr, out_ptr,
    n, k,
    strideL, stridex, strideo,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    for col in range(k):
        # Backward substitution using L^T
        for i in range(n - 1, -1, -1):
            val = tl.load(x_ptr + i * stridex + col)
            for j in range(i + 1, n):
                lt_val = tl.load(L_ptr + j * strideL + i)  # L[j,i] => L^T[i,j]
                out_val = tl.load(out_ptr + j * strideo + col)
                val -= lt_val * out_val
            diag = tl.load(L_ptr + i * strideL + i)
            val /= diag
            tl.store(out_ptr + i * strideo + col, val)

@triton.jit
def _cholesky_decomposition_kernel(
    A_ptr, n,
    strideA,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Naive Cholesky factorization (in-place in A)
    for i in range(n):
        # A[i,i] = sqrt(A[i,i] - sum(A[i,k]^2 for k < i))
        sum_val = 0.0
        for k in range(i):
            val = tl.load(A_ptr + i * strideA + k)
            sum_val += val * val
        diag_val = tl.load(A_ptr + i * strideA + i)
        diag_val = math.sqrt(diag_val - sum_val)
        tl.store(A_ptr + i * strideA + i, diag_val)

        # For j in i+1..n-1:
        for j in range(i + 1, n):
            sum_val_offdiag = 0.0
            for k in range(i):
                val_i_k = tl.load(A_ptr + i * strideA + k)
                val_j_k = tl.load(A_ptr + j * strideA + k)
                sum_val_offdiag += val_j_k * val_i_k
            a_j_i = tl.load(A_ptr + j * strideA + i)
            val_i_i = tl.load(A_ptr + i * strideA + i)
            updated = (a_j_i - sum_val_offdiag) / val_i_i
            tl.store(A_ptr + j * strideA + i, updated)
            # Zero upper triangle
            tl.store(A_ptr + i * strideA + j, 0.0)

def fused_cholesky_solve(A, b):
    """
    Computes the solution x to the equation Ax = b using Cholesky decomposition.
    A is overwritten with its Cholesky factor L on output.
    """
    n, _ = A.shape
    _, k = b.shape
    # Run Cholesky decomposition on A (in-place)
    grid = lambda meta: (1,)  # single-block dispatch for demonstration
    _cholesky_decomposition_kernel[grid](A, n, A.stride(0), BLOCK_SIZE=1)

    # Prepare x, same shape as b
    import torch
    x = torch.empty_like(b)
    # Forward substitution
    _forward_substitution_kernel[grid](
        A, b, x,
        n, k,
        A.stride(0), b.stride(0), x.stride(0),
        BLOCK_SIZE=1
    )
    # Backward substitution
    out = torch.empty_like(x)
    _backward_substitution_kernel[grid](
        A, x, out,
        n, k,
        A.stride(0), x.stride(0), out.stride(0),
        BLOCK_SIZE=1
    )
    return out
