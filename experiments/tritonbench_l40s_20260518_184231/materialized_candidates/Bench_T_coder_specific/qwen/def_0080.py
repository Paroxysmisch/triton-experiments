import triton
import triton.language as tl

@triton.jit
def qr_decomposition(A_ptr, m, n):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Perform QR decomposition using Householder method
    if row_idx < m:
        if col_idx <= row_idx:
            v = A_ptr[row_idx * n + col_idx]
            s = 0.0
            for i in range(row_idx + 1, m):
                s += A_ptr[i * n + col_idx] * A_ptr[i * n + col_idx]
            s = -tl.sqrt(s)
            if A_ptr[row_idx * n + col_idx] > 0:
                s = -s
            
            tau = s / (A_ptr[row_idx * n + col_idx] - s)
            c = 1.0 / (1.0 + tau * tau)
            s = c * tau
            
            A_ptr[row_idx * n + col_idx] -= s
            for j in range(col_idx + 1, n):
                A_ptr[row_idx * n + j] -= s * A_ptr[row_idx * n + j]
            
            for i in range(row_idx + 1, m):
                u = A_ptr[i * n + col_idx]
                v = A_ptr[i * n + col_idx]
                for j in range(col_idx + 1, n):
                    u -= s * A_ptr[row_idx * n + j]
                    v -= s * A_ptr[row_idx * n + j]
                A_ptr[i * n + col_idx] = u
                A_ptr[i * n + col_idx] = v
                
                A_ptr[i * n + col_idx] /= (1.0 + s * v)
                A_ptr[i * n + col_idx] /= (1.0 + s * v)

@triton.jit
def qr_solve(Q_ptr, R_ptr, b_ptr, m, n, k):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Compute Q^T b
    y = tl.zeros((n,), dtype=tl.float32)
    for i in range(n):
        sum_val = 0.0
        for j in range(m):
            sum_val += Q_ptr[j * n + i] * b_ptr[j * k + col_idx]
        y[i] = sum_val
    
    # Solve Rx = y using back substitution
    x = tl.zeros((n, k), dtype=tl.float32)
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += R_ptr[i * n + j] * x[j, col_idx]
        x[i, col_idx] = (y[i] - sum_val) / R_ptr[i * n + i]
    
    return x

@triton.jit
def fused_qr_solve(A_ptr, b_ptr, m, n, k):
    Q = tl.zeros((m, n), dtype=tl.float32)
    R = tl.zeros((n, n), dtype=tl.float32)
    
    # Copy A to Q
    for i in range(m):
        for j in range(n):
            Q[i * n + j] = A_ptr[i * n + j]
    
    # Perform QR decomposition
    qr_decomposition(Q, m, n)
    
    # Extract R from Q
    for i in range(n):
        for j in range(i, n):
            R[i * n + j] = Q[i * n + j]
    
    # Solve the linear system Rx = Q^T b
    x = qr_solve(Q, R, b_ptr, m, n, k)
    
    return x

# Example usage
m = 4
n = 3
k = 2
A = tl.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=tl.float32)
b = tl.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=tl.float32)

result = fused_qr_solve(A, b, m, n, k)
print(result)
