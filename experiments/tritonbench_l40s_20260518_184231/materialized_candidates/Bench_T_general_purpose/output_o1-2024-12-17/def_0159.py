import math
import torch
import triton
import triton.language as tl

@triton.jit
def _cholesky_step_kernel(
    A_ptr, L_ptr,
    n,
    k,
    strideA_row, strideA_col,
    strideL_row, strideL_col,
    is_complex: tl.constexpr,
    upper: tl.constexpr
):
    pid = tl.program_id(0)
    # Only one block processes the k-th step
    # 1) Compute L[k, k]
    diag_offset = k * strideA_row + k * strideA_col
    val = 0.
    if is_complex:
        # sum_{p=0}^{k-1} L[k,p]*conj(L[k,p])
        # We'll manually compute real part in val since L might be complex
        # Real part
        rp = 0.
        ip = 0.
        for p in range(k):
            l_re = tl.load(L_ptr + k * strideL_row + p * strideL_col)
            l_im = tl.load(L_ptr + k * strideL_row + p * strideL_col + 1)
            rp += l_re * l_re + l_im * l_im
        a_re = tl.load(A_ptr + diag_offset)
        a_im = tl.load(A_ptr + diag_offset + 1)
        # A is Hermitian => diagonal imaginary is zero, use real part
        re = a_re - rp
        re = tl.where(re > 0., re, 0.)  # clamp negative small values
        root = tl.sqrt(re)
        tl.store(L_ptr + k * strideL_row + k * strideL_col, root)
        tl.store(L_ptr + k * strideL_row + k * strideL_col + 1, 0.)
    else:
        # real
        for p in range(k):
            lp = tl.load(L_ptr + k * strideL_row + p * strideL_col)
            val += lp * lp
        a_val = tl.load(A_ptr + diag_offset)
        a_val = a_val - val
        a_val = tl.where(a_val > 0., a_val, 0.)
        root = tl.sqrt(a_val)
        tl.store(L_ptr + k * strideL_row + k * strideL_col, root)

    # 2) Compute L[i, k] for i = k+1..n-1 (if not upper)
    #    or L[k, i] for i = k+1..n-1 (if upper)
    for i in range(k + 1, n):
        if upper:
            # fill row k
            offA = k * strideA_row + i * strideA_col
            offLk = k * strideL_row + k * strideL_col
            offLi = k * strideL_row + i * strideL_col
        else:
            # fill column k
            offA = i * strideA_row + k * strideA_col
            offLk = k * strideL_row + k * strideL_col
            offLi = i * strideL_row + k * strideL_col
        val_re = 0.
        val_im = 0.
        if is_complex:
            a_re = tl.load(A_ptr + offA)
            a_im = tl.load(A_ptr + offA + 1)
            for p in range(k):
                if upper:
                    pl_re = tl.load(L_ptr + k * strideL_row + p * strideL_col)
                    pl_im = tl.load(L_ptr + k * strideL_row + p * strideL_col + 1)
                    ql_re = tl.load(L_ptr + i * strideL_row + p * strideL_col)
                    ql_im = tl.load(L_ptr + i * strideL_row + p * strideL_col + 1)
                else:
                    pl_re = tl.load(L_ptr + k * strideL_row + p * strideL_col)
                    pl_im = tl.load(L_ptr + k * strideL_row + p * strideL_col + 1)
                    ql_re = tl.load(L_ptr + i * strideL_row + p * strideL_col)
                    ql_im = tl.load(L_ptr + i * strideL_row + p * strideL_col + 1)
                # subtract L[i,p]*conj(L[k,p])
                # conj(L[k,p]) => pl_re, -pl_im
                val_re -= ql_re * pl_re + ql_im * (-pl_im)
                val_im -= ql_re * (-pl_im) - ql
