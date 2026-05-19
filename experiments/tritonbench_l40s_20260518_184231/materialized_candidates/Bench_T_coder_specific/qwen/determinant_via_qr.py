import triton
import triton.language as tl

@triton.jit
def qr_decomposition(A_ptr, Q_ptr, R_ptr, n):
    pid = tl.program_id(0)
    row_start = pid * n
    row_end = min(row_start + n, n * n)

    # Initialize Q and R
    for i in range(n):
        tl.store(Q_ptr + row_start * n + i, tl.zeros_like(tl.load(A_ptr + row_start * n + i)))
        tl.store(R_ptr + row_start * n + i, tl.zeros_like(tl.load(A_ptr + row_start * n + i)))

    for k in range(row_start // n, row_end // n):
        x_norm = tl.norm(tl.load(A_ptr + row_start * n + k), ord=2)
        v = tl.load(A_ptr + row_start * n + k)
        v[0] += tl.where(v[0] >= 0, x_norm, -x_norm)
        v_norm = tl.norm(v, ord=2)
        v /= v_norm

        # Apply Householder transformation to A
        for j in range(k + 1, row_end // n):
            alpha = tl.dot(v, tl.load(A_ptr + row_start * n + j))
            beta = -alpha / v_norm**2
            for i in range(n):
                tl.atomic_add(A_ptr + row_start * n + i + j * n, beta * v[i])

        # Update Q and R
        for i in range(n):
            q_i_k = tl.dot(v, tl.load(Q_ptr + row_start * n + i))
            r_i_k = tl.dot(tl.load(Q_ptr + row_start * n + i), v)
            tl.atomic_add(Q_ptr + row_start * n + i + k * n, q_i_k)
            tl.atomic_add(R_ptr + row_start * n + i + k * n, r_i_k)

# Triton wrapper function
@triton.jit
def determinant_via_qr_kernel(A_ptr, det_ptr, n):
    pid = tl.program_id(0)
    row_start = pid * n
    row_end = min(row_start + n, n * n)

    Q = tl.zeros((n, n), dtype=A.dtype)
    R = tl.zeros((n, n), dtype=A.dtype)

    # Perform QR decomposition
    qr_decomposition(A_ptr, Q.ptr, R.ptr, n)

    # Calculate determinant from R
    det = tl.ones_like(tl.load(A_ptr))
    for i in range(n):
        det *= tl.load(R.ptr + row_start * n + i + i * n)

    # Store determinant
    tl.store(det_ptr, det)

# Wrapper function
def determinant_via_qr(A, *, mode='reduced', out=None):
    n = A.shape[0]
    assert A.shape == (n, n), "Matrix must be square"
    
    if out is None:
        out = tl.zeros((1,), dtype=A.dtype)
    
    # Allocate memory for Q and R
    Q = tl.zeros((n, n), dtype=A.dtype)
    R = tl.zeros((n, n), dtype=A.dtype)
    
    # Call Triton kernel
    determinant_via_qr_kernel[A.block_size](A.ptr, out.ptr, n)
    
    return out
