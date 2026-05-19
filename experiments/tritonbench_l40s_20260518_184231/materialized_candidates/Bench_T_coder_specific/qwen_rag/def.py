import triton
import triton.language as tl
import torch

# Define constants for Triton compilation
BLOCK_SIZE = 32

@triton.jit
def lu_kernel(A_ptr, P_ptr, L_ptr, U_ptr, M, N, stride_A, stride_P, stride_L, stride_U, diagonal, pivot):
    row = tl.program_id(0)
    col = tl.program_id(1)
    m_mask = row < M
    n_mask = col < N
    mask = m_mask & n_mask

    a = tl.load(A_ptr + row * stride_A + col, mask=mask)
    
    if row == col:
        u = a
        l = 1.0
        p = 0
    else:
        u = 0.0
        l = 0.0
        p = 0
        
        if row > col:
            # Forward elimination
            l = a / u
            a -= l * U_ptr[row * stride_U + col]
            
            # Update L
            tl.store(L_ptr + row * stride_L + col, l, mask=mask)
            
            # Update U
            U_ptr[row * stride_U + col] = a
    
    tl.store(U_ptr + row * stride_U + col, u, mask=mask)
    
    if pivot:
        if row > col:
            # Find maximum pivot
            max_row = tl.max(tl.where(P_ptr + row * stride_P + col == 0, -1, row))
            if max_row != row:
                # Swap rows
                temp = A_ptr[row * stride_A + col]
                A_ptr[row * stride_A + col] = A_ptr[max_row * stride_A + col]
                A_ptr[max_row * stride_A + col] = temp
                
                temp = L_ptr[row * stride_L + col]
                L_ptr[row * stride_L + col] = L_ptr[max_row * stride_L + col]
                L_ptr[max_row * stride_L + col] = temp
                
                temp = U_ptr[row * stride_U + col]
                U_ptr[row * stride_U + col] = U_ptr[max_row * stride_U + col]
                U_ptr[max_row * stride_U + col] = temp
                
                temp = P_ptr[row * stride_P + col]
                P_ptr[row * stride_P + col] = P_ptr[max_row * stride_P + col]
                P_ptr[max_row * stride_P + col] = temp
    
    tl.store(A_ptr + row * stride_A + col, a, mask=mask)

@triton.jit
def solve_lower_kernel(L_ptr, B_ptr, Y_ptr, M, K, stride_L, stride_B, stride_Y):
    row = tl.program_id(0)
    col = tl.program_id(1)
    m_mask = row < M
    k_mask = col < K
    mask = m_mask & k_mask

    b = tl.load(B_ptr + row * stride_B + col, mask=mask)
    
    if row == 0:
        y = b / L_ptr[row * stride_L + row]
    else:
        y = b
        for i in range(row):
            y -= L_ptr[row * stride_L + i] * Y_ptr[i * stride_Y + col]
        
    tl.store(Y_ptr + row * stride_Y + col, y, mask=mask)

@triton.jit
def solve_upper_kernel(U_ptr, Y_ptr, X_ptr, M, K, stride_U, stride_Y, stride_X):
    row = tl.program_id(0)
    col = tl.program_id(1)
    m_mask = row < M
    k_mask = col < K
    mask = m_mask & k_mask

    y = tl.load(Y_ptr + row * stride_Y + col, mask=mask)
    
    if row == M - 1:
        x = y / U_ptr[row * stride_U + row]
    else:
        x = y
        for i in range(M - 1, row, -1):
            x -= U_ptr[row * stride_U + i] * X_ptr[i * stride_X + col]
        
    tl.store(X_ptr + row * stride_X + col, x, mask=mask)

def solve_multiple_lu(A, Bs, *, pivot=True, out=None) -> Tensor:
    A = A.contiguous()
    Bs = Bs.contiguous()
    M, N = A.shape[-2:]
    K = Bs.shape[-1]
    assert M == N, "Matrix A must be square"
    
    if out is None:
        out = torch.empty_like(Bs)
    
    with torch.cuda.device(A.device):
        # Allocate memory for LU decomposition
        L = torch.zeros_like(A)
        U = torch.zeros_like(A)
        P = torch.eye(M, dtype=A.dtype, device=A.device)
        
        # Perform LU decomposition
        grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
        lu_kernel[grid](A.data_ptr(), P.data_ptr(), L.data_ptr(), U.data_ptr(), M, N, A.stride(0), P.stride(0), L.stride(0), U.stride(0), 0, pivot)
        
        # Solve linear systems
        for k in range(K):
            B_k = Bs[..., k]
            Y_k = torch.empty_like(B_k)
            X_k = torch.empty_like(B_k)
            
            # Solve Ly = Pb
            grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(K, BLOCK_SIZE))
            solve_lower_kernel[L.grid()](U.data_ptr(), B_k.data_ptr(), Y_k.data_ptr(), M, K, U.stride(0), B_k.stride(0), Y_k.stride(0))
            
            # Solve Ux = y
            grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(K, BLOCK_SIZE))
            solve_upper_kernel[U.data_ptr(), Y_k.data_ptr(), X_k.data_ptr(), M, K, U.stride(0), Y_k.stride(0), X_k.stride(0)]
            
            out[..., k] = X_k
    
    return out
