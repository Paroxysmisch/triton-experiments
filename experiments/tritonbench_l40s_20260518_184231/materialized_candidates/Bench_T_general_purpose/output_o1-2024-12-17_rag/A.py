import torch
import triton
import triton.language as tl

@triton.jit
def _solve_left_no_pivot_kernel(
    A_ptr, B_ptr, out_ptr,
    batch_strideA, batch_strideB, batch_strideOut,
    strideA_row, strideA_col,
    strideB_row, strideB_col,
    strideOut_row, strideOut_col,
    N, NRHS,
    BLOCK_SIZE: tl.constexpr
):
    """
    Naive Gauss elimination + back-substitution without pivoting (left solve: AX = B).
    One program handles one batch of A and B simultaneously (single-block approach).

    This kernel assumes:
      - A is (N x N).
      - B is (N x NRHS).
      - out will be written with the solution (N x NRHS).
      - No pivoting, so diagonal elements are assumed non-zero.
      - Each thread block handles one entire batch entry.
    """

    batch_id = tl.program_id(0)

    # Offsets for this batch
    A_base = A_ptr + batch_id * batch_strideA
    B_base = B_ptr + batch_id * batch_strideB
    Out_base = out_ptr + batch_id * batch_strideOut

    # Forward Elimination (in-place on B, treat A as read-only)
    for k in range(N):
        pivot_val = tl.load(A_base + k * strideA_row + k * strideA_col)
        # Divide row k of B by pivot_val
        # Each column of B is handled in a loop (since we do not launch multiple threads for columns).
        for colB in range(NRHS):
            b_elem = tl.load(B_base + k * strideB_row + colB * strideB_col)
            b_elem = b_elem / pivot_val
            tl.store(B_base + k * strideB_row + colB * strideB_col, b_elem)

        # Eliminate below
        for row in range(k + 1, N):
            factor = tl.load(A_base + row * strideA_row + k * strideA_col) / pivot_val
            for colB in range(NRHS):
                b_k = tl.load(B_base + k * strideB_row + colB * strideB_col)
                b_row = tl.load(B_base + row * strideB_row + colB * strideB_col)
                tl.store(B_base + row * strideB_row + colB * strideB_col, b_row - factor * b_k)

    # Back Substitution
    for k in range(N - 1, -1, -1):
        for colB in range(NRHS):
            sum_ = tl.load(B_base + k * strideB_row + colB * strideB_col)
            # subtract known upper portion
            for colA in range(k + 1, N):
                a_kcol = tl.load(A_base + k * strideA_row + colA * strideA_col)
                b_colA = tl.load(Out_base + colA * strideOut_row + colB * strideOut_col)
                sum_ -= a_kcol * b_colA
            tl.store(Out_base + k * strideOut_row + colB * strideOut_col, sum_)

@triton.jit
def _copy_matrix_kernel(
    dst_ptr, src_ptr,
    stride_dst_row, stride_dst_col,
    stride_src_row, stride_src_col,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    """
    Simple 2D copy kernel. Each program copies a tile [BLOCK_M x BLOCK_N].
    Used for copying B into out if out is provided or for in-kernel usage.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create 2D mesh
    rm = row_offset[:, None]
    cn = col_offset[None, :]

    # Flatten index for src/dst
    dst_indices = rm * stride_dst_row + cn * stride_dst_col
    src_indices = rm * stride_src_row + cn * stride_src_col

    # Boundary mask
    mask = (rm < M) & (cn < N)

    val = tl.load(src_ptr + src_indices, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_indices, val, mask=mask)

def solve(A, B, *, left=True, out=None):
    """
    Triton-based solver for square systems:
      AX = B if left=True
      XA = B if left=False

    Supports batches, in which case A.shape = (..., N, N), B.shape = (..., N, NRHS) [left=True]
    or B.shape = (..., NRHS, N) [left=False], same leading batch dims.

    This function synchronizes the current CUDA stream with the CPU.
    NOTE: This is a naive reference version without pivoting, not recommended for production use.
    """

    # Basic checks
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("Tensors must be on CUDA device for Triton-based solve.")
    if left:
        # A: (..., N, N), B: (..., N, NRHS)
        if A.shape[-1] != A.shape[-2]:
            raise ValueError("A must be square for left solve.")
        if A.shape[-2] != B.shape[-2]:
            raise ValueError("A and B must have compatible shapes for left solve (A NxN, B NxNRHS).")
        N = A.shape[-1]
        NRHS = B.shape[-1]
        batch_dims = A.shape[:-2]
    else:
        # Solve XA = B => X = B A^-1 if A is NxN, then B: (..., NRHS, N)
        # For simplicity, let us transpose the problem to mimic left-solve:
        #   (A^T X^T = B^T) => X^T = A^-T B^T => X = (X^T)^T
        # We'll do a quick check or just bail out with ValueError for a mismatch
        if A.shape[-1] != A.shape[-2]:
            raise ValueError("A must be square for right solve.")
        if A.shape[-2] != B.shape[-1]:
            raise ValueError("For right solve, B shape must be (..., NRHS, N) so that B.shape[-1] == N.")
        N = A.shape[-1]
        NRHS = B.shape[-2]
        batch_dims = A.shape[:-2]

    # Create out if needed
    # For left solve, out is (same batch..., N, NRHS)
    # For right solve, out is (same batch..., NRHS, N), but we handle that by a transpose approach below
    if out is None:
        if left:
            out = torch.empty_like(B)
        else:
            # For a right solve, we expect X to have shape (..., NRHS, N)
            out = torch.empty_like(B)

    # If left: we can directly run the naive solve.
    # If right: we will do a transpose trick for A and B, solve left, then transpose back.
    if not left:
        # Transpose A to shape (..., N, N):
        A_t = A.mT
        # Transpose B to shape (..., N, NRHS):
        B_t = B.mT
        # out_t will be shape (..., N, NRHS) after solve
        out
