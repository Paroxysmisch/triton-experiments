import torch
import triton
import triton.language as tl

# Triton kernel for QR decomposition and solving the linear system
@triton.jit
def fused_qr_solve_kernel(A, b, x, m, n, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Load A and b into shared memory
    A_shared = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    b_shared = tl.zeros((BLOCK_SIZE, k), dtype=tl.float32)

    # Load A and b into shared memory
    for i in range(BLOCK_SIZE):
        if block_start + i < m:
            for j in range(n):
                A_shared[i, j] = A[block_start + i, j]
            for j in range(k):
                b_shared[i, j] = b[block_start + i, j]

    # Perform QR decomposition on A
    Q, R = tl.qr(A_shared)

    # Compute Q^T b
    Q_T = tl.transpose(Q)
    Q_T_b = tl.dot(Q_T, b_shared)

    # Solve R x = Q^T b
    x_shared = tl.solve_triangular(R, Q_T_b, upper=True)

    # Store the result back to x
    for i in range(BLOCK_SIZE):
        if block_start + i < m:
            for j in range(k):
                x[block_start + i, j] = x_shared[i, j]

# Wrapper function to invoke the Triton kernel
def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = A.shape
    _, k = b.shape
    assert m >= n, "Matrix A must have m >= n"

    # Prepare output tensor x with the same shape and type as b
    x = torch.empty_like(b)

    # Determine the optimal block size
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(m)))

    # Determine the grid size needed to cover all elements
    grid_size = triton.cdiv(m, block_size)

    # Launch the Triton kernel
    fused_qr_solve_kernel[(grid_size, 1, 1)](A, b, x, m, n, k, block_size)

    return x
