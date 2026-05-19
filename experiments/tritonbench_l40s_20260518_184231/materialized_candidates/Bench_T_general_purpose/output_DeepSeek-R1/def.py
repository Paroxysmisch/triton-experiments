import torch
import triton
import triton.language as tl

@triton.jit
def forward_substitution_kernel(
    L_ptr, B_ptr, Y_ptr,
    n, k,
    stride_L_batch, stride_L_row, stride_L_col,
    stride_B_batch, stride_B_row, stride_B_rhs,
    stride_Y_batch, stride_Y_row, stride_Y_rhs,
    total_problems,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= total_problems:
        return
    batch_rhs = pid
    batch_idx = batch_rhs // k
    rhs_idx = batch_rhs % k

    # Iterate over each row in sequence
    for i in range(n):
        sum_val = 0.0
        # Compute sum(L[i, j] * Y[j] for j < i)
        for j in range(i):
            L_ij = tl.load(L_ptr + batch_idx * stride_L_batch + i * stride_L_row + j * stride_L_col)
            Y_j = tl.load(Y_ptr + batch_idx * stride_Y_batch + j * stride_Y_row + rhs_idx * stride_Y_rhs)
            sum_val += L_ij * Y_j
        # Y[i] = B[i] - sum_val
        B_i = tl.load(B_ptr + batch_idx * stride_B_batch + i * stride_B_row + rhs_idx * stride_B_rhs)
        Y_i = B_i - sum_val
        tl.store(Y_ptr + batch_idx * stride_Y_batch + i * stride_Y_row + rhs_idx * stride_Y_rhs, Y_i)

@triton.jit
def backward_substitution_kernel(
    U_ptr, Y_ptr, X_ptr,
    n, k,
    stride_U_batch, stride_U_row, stride_U_col,
    stride_Y_batch, stride_Y_row, stride_Y_rhs,
    stride_X_batch, stride_X_row, stride_X_rhs,
    total_problems,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= total_problems:
        return
    batch_rhs = pid
    batch_idx = batch_rhs // k
    rhs_idx = batch_rhs % k

    # Iterate from the last row upwards
    for i in range(n-1, -1, -1):
        sum_val = 0.0
        # Compute sum(U[i, j] * X[j] for j > i)
        for j in range(i+1, n):
            U_ij = tl.load(U_ptr + batch_idx * stride_U_batch + i * stride_U_row + j * stride_U_col)
            X_j = tl.load(X_ptr + batch_idx * stride_X_batch + j * stride_X_row + rhs_idx * stride_X_rhs)
            sum_val += U_ij * X_j
        # X[i] = (Y[i] - sum_val) / U[i, i]
        U_ii = tl.load(U_ptr + batch_idx * stride_U_batch + i * stride_U_row + i * stride_U_col)
        Y_i = tl.load(Y_ptr + batch_idx * stride_Y_batch + i * stride_Y_row + rhs_idx * stride_Y_rhs)
        X_i = (Y_i - sum_val) / U_ii
        tl.store(X_ptr + batch_idx * stride_X_batch + i * stride_X_row + rhs_idx * stride_X_rhs, X_i)

def solve_multiple_lu(A: torch.Tensor, Bs: torch.Tensor, *, pivot=True, out=None) -> torch.Tensor:
    assert A.shape[:-2] == Bs.shape[:-2], "Batch dimensions of A and Bs must match"
    assert A.size(-1) == A.size(-2), "A must be square"
    assert A.size(-1) == Bs.size(-2), "Shapes of A and Bs must match"

    # Compute LU decomposition
    LU, pivots = torch.lu(A, pivot=pivot)
    P, L, U = torch.lu_unpack(LU, pivots)

    # Apply permutation to Bs
    if pivot:
        perm_indices = torch.argmax(P, dim=-1)
        perm_indices_expanded = perm_indices.unsqueeze(-1).expand(*Bs.shape)
        Bs_permuted = torch.gather(Bs, dim=-2, index=perm_indices_expanded)
    else:
        Bs_permuted = Bs

    # Flatten batch dimensions for Triton kernels
    original_batch_shape = Bs_permuted.shape[:-2]
    n = Bs_permuted.size(-2)
    k = Bs_permuted.size(-1)
    L_flat = L.contiguous().view(-1, n, n)
    Bs_flat = Bs_permuted.contiguous().view(-1, n, k)
    total_batches = L_flat.size(0)
    total_problems = total_batches * k

    # Allocate output tensor for forward substitution
    Y = torch.empty_like(Bs_flat)

    # Launch forward substitution kernel
    grid = lambda meta: (total_problems,)
    forward_substitution_kernel[grid](
        L_flat, Bs_flat, Y,
        n, k,
        L_flat.stride(0), L_flat.stride(1), L_flat.stride(2),
        Bs_flat.stride(0), Bs_flat.stride(1), Bs_flat.stride(2),
        Y.stride(0), Y.stride(1), Y.stride(2),
        total_problems,
        BLOCK_SIZE=32,
    )

    # Allocate output tensor for backward substitution
    X = torch.empty_like(Y)

    # Launch backward substitution kernel
    backward_substitution_kernel[grid](
        U.contiguous().view(-1, n, n), Y, X,
        n, k,
        U.stride(0), U.stride(1), U.stride(2),
        Y.stride(0), Y.stride(1), Y.stride(2),
        X.stride(0), X.stride(1), X.stride(2),
        total_problems,
        BLOCK_SIZE=32,
    )

    # Reshape to original batch dimensions
    X = X.view(*original_batch_shape, n, k)

    if out is not None:
        out.copy_(X)
    return X
