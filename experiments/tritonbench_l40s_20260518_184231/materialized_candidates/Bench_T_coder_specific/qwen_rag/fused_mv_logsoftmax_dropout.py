@triton.jit
def mv(mat, vec, out, N: tl.constexpr):
    """Performs matrix-vector multiplication."""
    pid = tl.program_id(axis=0)
    x_coord = pid % mat.shape[0]
    y_coord = pid // mat.shape[0]
    
    acc = 0.0
    for k in range(N):
        acc += mat[x_coord, k] * vec[k]
    
    out[x_coord, y_coord] = acc
