import torch
import triton
import triton.language as tl

@triton.jit
def lower_triangular_solve_kernel(
    L_ptr, B_ptr, Y_ptr, n,
    stride_L_batch, stride_L_row, stride_L_col,
    stride_B_batch, stride_B_row, stride_B_col,
    stride_Y_batch, stride_Y_row, stride_Y_col,
    BLOCK_SIZE: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            l_ij = tl.load(L_ptr + pid_batch * stride_L_batch + i * stride_L_row + j * stride_L_col)
            y_jk = tl.load(Y_ptr + pid_batch * stride_Y_batch + j * stride_Y_row + pid_col * stride_Y_col)
            sum_val += l_ij * y_jk
        b_ik = tl.load(B_ptr + pid_batch * stride_B_batch + i * stride_B_row + pid_col * stride_B_col)
        y_ik = b_ik - sum_val
        tl.store(Y_ptr + pid_batch * stride_Y_batch + i * stride_Y_row + pid_col * stride_Y_col, y_ik)

@triton.jit
def upper_triangular_solve_kernel(
    U_ptr, Y_ptr, X_ptr, n,
    stride_U_batch, stride_U_row, stride_U_col,
    stride_Y_batch, stride_Y_row, stride_Y_col,
    stride_X_batch, stride_X_row, stride_X_col,
    BLOCK_SIZE: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    for i in range(n-1, -1, -1):
        sum_val = 0.0
        for j in range(i+1, n):
            u_ij = tl.load(U_ptr + pid_batch * stride_U_batch + i * stride_U_row + j * stride_U_col)
            x_jk = tl.load(X_ptr + pid_batch * stride_X_batch + j * stride_X_row + pid_col * stride_X_col)
            sum_val += u_ij * x_jk
        y_ik = tl.load(Y_ptr + pid_batch * stride_Y_batch + i * stride_Y_row + pid_col * stride_Y_col)
        u_ii = tl.load(U_ptr + pid_batch * stride_U_batch + i * stride_U_row + i * stride_U_col)
        x_ik = (y_ik - sum_val) / u_ii
        tl.store(X_ptr + pid_batch * stride_X_batch + i * stride_X_row + pid_col * stride_X_col, x_ik)

def invert_matrix_lu(A: torch.Tensor, *, pivot=True, out=None) -> torch.Tensor:
    assert A.shape[-1] == A.shape[-2], "A must be a square matrix or batches of square matrices"
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError(f"Unsupported dtype {A.dtype}")
    
    batch_dims = A.shape[:-2]
    n = A.size(-1)
    batch_size = int(torch.prod(torch.tensor(batch_dims))) if batch_dims else 1
    
    if pivot:
        LU, pivots = torch.lu(A, get_infos=True)
        P, L, U = torch.lu_unpack(LU, pivots)
    else:
        raise NotImplementedError("LU without pivoting is not implemented in this solution")
    
    I = torch.eye(n, dtype=A.dtype, device=A.device).expand_as(A)
    B = P @ I
    
    Y = torch.empty_like(B)
    grid = (batch_size, n)
    L = L.contiguous()
    B = B.contiguous()
    Y = Y.contiguous()
    
    lower_triangular_solve_kernel[grid](
        L, B, Y, n,
        L.stride(-3) if L.ndim > 2 else 0, L.stride(-2), L.stride(-1),
        B.stride(-3) if B.ndim > 2 else 0, B.stride(-2), B.stride(-1),
        Y.stride(-3) if Y.ndim > 2 else 0, Y.stride(-2), Y.stride(-1),
        BLOCK_SIZE=128
    )
    
    X = torch.empty_like(Y)
    U = U.contiguous()
    X = X.contiguous()
    
    upper_triangular_solve_kernel[grid](
        U, Y, X, n,
        U.stride(-3) if U.ndim > 2 else 0, U.stride(-2), U.stride(-1),
        Y.stride(-3) if Y.ndim > 2 else 0, Y.stride(-2), Y.stride(-1),
        X.stride(-3) if X.ndim > 2 else 0, X.stride(-2), X.stride(-1),
        BLOCK_SIZE=128
    )
    
    if out is not None:
        out.copy_(X)
    return X
