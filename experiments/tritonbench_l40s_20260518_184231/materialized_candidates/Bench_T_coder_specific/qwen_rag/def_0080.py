import triton
import triton.language as tl

# Constants
BLOCK_SIZE = 64

@triton.jit
def qr_decompose_kernel(A_ptr, Q_ptr, R_ptr, m, n):
    row = tl.program_id(0)
    col = tl.program_id(1)

    if row >= m or col >= n:
        return

    if row == col:
        a_row_col = tl.load(A_ptr + row * n + col)
        sum_sq = a_row_col * a_row_col
        for i in range(row + 1, m):
            a_i_col = tl.load(A_ptr + i * n + col)
            sum_sq += a_i_col * a_i_col
        q_row_col = a_row_col / tl.sqrt(sum_sq)
        r_row_col = tl.sqrt(sum_sq)
        tl.store(Q_ptr + row * n + col, q_row_col)
        tl.store(R_ptr + row * n + col, r_row_col)
    else:
        q_row_col = tl.zeros((), dtype=tl.float32)
        r_row_col = tl.zeros((), dtype=tl.float32)
        for i in range(col, m):
            a_i_col = tl.load(A_ptr + i * n + col)
            q_row_col += a_i_col * tl.load(Q_ptr + row * n + i)
        q_row_col /= tl.load(R_ptr + col * n + col)
        for i in range(col, m):
            a_i_col = tl.load(A_ptr + i * n + col)
            tl.atomic_add(R_ptr + row * n + i, a_i_col * q_row_col)
            for j in range(i + 1, m):
                a_j_col = tl.load(A_ptr + j * n + col)
                tl.atomic_add(A_ptr + j * n + i, a_j_col * q_row_col)
        tl.store(Q_ptr + row * n + col, q_row_col)
        tl.store(R_ptr + row * n + col, R_ptr[col * n + col])

@triton.jit
def forward_substitution_kernel(R_ptr, b_ptr, x_ptr, m, n):
    row = tl.program_id(0)
    col = tl.program_id(1)

    if row >= m or col >= n:
        return

    if row == col:
        r_row_col = tl.load(R_ptr + row * n + col)
        b_row_col = tl.load(b_ptr + row)
        x_row_col = b_row_col / r_row_col
        tl.store(x_ptr + row, x_row_col)
    else:
        r_row_col = tl.load(R_ptr + row * n + col)
        x_row_col = tl.load(x_ptr + row)
        for i in range(col, m):
            r_i_col = tl.load(R_ptr + i * n + col)
            x_i_col = tl.load(x_ptr + i)
            tl.atomic_add(x_ptr + row, -r_i_col * x_i_col)
        x_row_col /= r_row_col
        tl.store(x_ptr + row, x_row_col)

def fused_qr_solve(A: tl.tensor, b: tl.tensor) -> tl.tensor:
    m, n = A.shape
    k = b.shape[1]
    
    # Allocate memory for Q, R, and X
    Q = tl.zeros((m, n), dtype=A.dtype)
    R = tl.zeros((n, n), dtype=A.dtype)
    X = tl.zeros((n, k), dtype=A.dtype)
    
    # Perform QR decomposition
    qr_decompose_kernel[(m, n)](A, Q, R, m, n)
    
    # Solve Rx = Q^T b using forward substitution
    Q_T_b = tl.dot(Q.T, b)
    forward_substitution_kernel[(n, k)](R, Q_T_b, X, n, k)
    
    return X
