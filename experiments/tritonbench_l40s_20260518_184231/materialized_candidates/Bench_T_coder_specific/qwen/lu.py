import triton
import triton.language as tl

# Define the Triton kernel for LU decomposition with partial pivoting
@triton.jit
def lu_kernel(
    A_ptr, P_ptr, L_ptr, U_ptr,
    m, n, batch_size, stride_A, stride_P, stride_L, stride_U,
    pid=triton.program_id(0), block_size=32):
    
    # Each thread handles one element of U
    row = pid % m
    col = pid // m
    
    if row >= n or col >= n:
        return
    
    # Compute L[row, col] for all rows <= col
    if row <= col:
        sum = 0.0
        for k in range(row):
            sum += L_ptr[tl.arange(row) * stride_L + k * stride_L + col]
        U_ptr[row * stride_U + col] = A_ptr[row * stride_A + col] - sum
    
    # Compute U[row, col] for all rows > col
    else:
        sum = 0.0
        for k in range(col):
            sum += L_ptr[row * stride_L + k * stride_L + col] * U_ptr[k * stride_U + col]
        U_ptr[row * stride_U + col] = A_ptr[row * stride_A + col] - sum
    
    # Pivot if necessary
    if row == col and pivot:
        max_val = abs(U_ptr[row * stride_U + col])
        max_idx = row
        for i in range(row+1, m):
            val = abs(U_ptr[i * stride_U + col])
            if val > max_val:
                max_val = val
                max_idx = i
        
        if max_idx != row:
            # Swap rows in A
            for j in range(n):
                temp = A_ptr[row * stride_A + j]
                A_ptr[row * stride_A + j] = A_ptr[max_idx * stride_A + j]
                A_ptr[max_idx * stride_A + j] = temp
            
            # Swap rows in P
            for j in range(m):
                temp = P_ptr[row * stride_P + j]
                P_ptr[row * stride_P + j] = P_ptr[max_idx * stride_P + j]
                P_ptr[max_idx * stride_P + j] = temp
    
    # Compute L[row, col] for all rows > col
    if row > col:
        sum = 0.0
        for k in range(col):
            sum += L_ptr[row * stride_L + k * stride_L + col] * U_ptr[k * stride_U + col]
        L_ptr[row * stride_L + col] = (A_ptr[row * stride_A + col] - sum) / U_ptr[col * stride_U + col]

# Define the Triton wrapper function
def lu(A, *, pivot=True, out=None):
    assert A.ndim >= 2 and A.shape[-2:] == (m, n), "Input must be a tensor of shape (*, m, n)"
    batch_size = A.shape[:-2]
    m, n = A.shape[-2:]
    
    # Allocate memory for outputs
    if pivot:
        P = torch.empty_like(A)
    else:
        P = torch.empty((batch_size, m, n), dtype=A.dtype, device=A.device)
    
    L = torch.empty_like(A)
    U = torch.empty_like(A)
    
    # Launch the kernel
    grid = (triton.cdiv(m*n, 32), len(batch_size))
    block = 32
    
    lu_kernel[grid, block](
        A.data_ptr(), P.data_ptr() if pivot else None, L.data_ptr(), U.data_ptr(),
        m, n, len(batch_size), A.stride(-1), P.stride(-1) if pivot else 0, L.stride(-1), U.stride(-1),
        pivot=pivot
    )
    
    if out is not None:
        out[0], out[1], out[2] = P, L, U
    
    return P, L, U
