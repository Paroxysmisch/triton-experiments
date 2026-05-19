import torch
import triton
import triton.language as tl

@triton.jit
def forward_sub_kernel(
    L_ptr, Pb_ptr, y_ptr,
    n: int, stride_L_row: int, stride_L_col: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= 1:
        return
    y = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            l_ij = tl.load(L_ptr + i * stride_L_row + j * stride_L_col)
            sum_val += l_ij * y[j]
        pb_i = tl.load(Pb_ptr + i)
        y_i = pb_i - sum_val
        y = tl.store(y, i, y_i)
    for i in range(n):
        tl.store(y_ptr + i, y[i])

@triton.jit
def backward_sub_kernel(
    U_ptr, y_ptr, x_ptr,
    n: int, stride_U_row: int, stride_U_col: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= 1:
        return
    x = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(n-1, -1, -1):
        sum_val = 0.0
        for j in range(i+1, n):
            u_ij = tl.load(U_ptr + i * stride_U_row + j * stride_U_col)
            sum_val += u_ij * x[j]
        y_i = tl.load(y_ptr + i)
        u_ii = tl.load(U_ptr + i * stride_U_row + i * stride_U_col)
        x_i = (y_i - sum_val) / u_ii
        x = tl.store(x, i, x_i)
    for i in range(n):
        tl.store(x_ptr + i, x[i])

def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 2 and A.size(0) == A.size(1), "A must be a square matrix"
    n = A.size(0)
    assert b.size() == (n,), "b must be a vector of size n"
    
    # Compute LU decomposition
    A_lu, pivots = torch.lu(A, pivot=True)
    P, L, U = torch.lu_unpack(A_lu, pivots)
    
    # Apply permutation to b
    Pb = torch.matmul(P, b)
    
    # Ensure data is in float32 and contiguous
    L = L.contiguous().to(torch.float32)
    U = U.contiguous().to(torch.float32)
    Pb = Pb.contiguous().to(torch.float32)
    y = torch.empty_like(Pb)
    x = torch.empty_like(y)
    
    # Launch forward substitution
    grid = lambda meta: (1,)
    BLOCK_SIZE = triton.next_power_of_2(n)
    forward_sub_kernel[grid](
        L, Pb, y,
        n, L.stride(0), L.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Launch backward substitution
    backward_sub_kernel[grid](
        U, y, x,
        n, U.stride(0), U.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x
