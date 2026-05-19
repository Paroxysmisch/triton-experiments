import triton
import triton.language as tl

@triton.jit
def qr_decomposition_kernel(
    A_ptr,
    Q_ptr,
    R_ptr,
    m,
    n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(m, BLOCK_SIZE_M)
    grid_n = tl.cdiv(n, BLOCK_SIZE_N)
    
    row = pid // grid_n
    col = pid % grid_n
    
    if row >= m or col >= n:
        return
    
    # Initialize Q and R
    q = tl.tensor([0.0] * n, dtype=tl.float32)
    r = tl.tensor([0.0] * n, dtype=tl.float32)
    
    for i in range(row, m):
        a_i = tl.load(A_ptr + i * n + row)
        r[row] += a_i * a_i
        if col == row:
            q[row] = a_i / tl.sqrt(r[row])
            tl.store(Q_ptr + i * n + row, q[row])
        elif col > row:
            r[col] += a_i * q[row]
            tl.store(R_ptr + row * n + col, r[col])
    
    for j in range(col + 1, n):
        s = 0.0
        for i in range(row, m):
            a_i = tl.load(A_ptr + i * n + row)
            s += a_i * q[j]
        
        for i in range(row, m):
            a_i = tl.load(A_ptr + i * n + row)
            a_j = tl.load(A_ptr + i * n + j)
            tl.atomic_add(A_ptr + i * n + row, -s * q[j])
            tl.atomic_add(A_ptr + i * n + j, -s * q[row])

def invoke_qr_decomposition_kernel(
    A: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    m: int,
    n: int,
    config: Dict[str, Any],
):
    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n, META["BLOCK_SIZE_N"]),
    )
    
    qr_decomposition_kernel[grid](
        A,
        Q,
        R,
        m,
        n,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,
        **config,
    )
