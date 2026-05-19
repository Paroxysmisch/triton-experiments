import triton
import triton.language as tl

# Constants for block sizes
BLOCK_N = 128
BLOCK_M = 32

@triton.jit
def mv_kernel(
    A_ptr,  # Pointer to matrix A of size N x M
    B_ptr,  # Pointer to vector B of size M
    C_ptr,  # Pointer to output vector C of size N
    N,      # Number of rows in matrix A
    M,      # Number of columns in matrix A and size of vector B
    stride_A_n,  # Stride of matrix A in the N dimension
    stride_A_m,  # Stride of matrix A in the M dimension
    stride_C_n,  # Stride of output vector C in the N dimension
    BLOCK_N: tl.constexpr,  # Block size for N
    BLOCK_M: tl.constexpr   # Block size for M
):
    # Compute the row index for the current block
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_N
    row_end = row_start + BLOCK_N

    # Load the vector B into shared memory
    B = tl.load(B_ptr, mask=tl.arange(0, BLOCK_M) < M)

    # Initialize the output vector C for the current block
    C = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Iterate over the matrix A in blocks of BLOCK_M
    for m in range(0, M, BLOCK_M):
        # Load the sub-matrix A of size BLOCK_N x BLOCK_M
        A = tl.load(A_ptr + row_start * stride_A_n + m * stride_A_m, 
                    mask=(tl.arange(0, BLOCK_N)[:, None] < N) & (tl.arange(0, BLOCK_M) < M))

        # Perform element-wise multiplication and accumulation
        C += tl.dot(A, B[m : m + BLOCK_M])

    # Store the results in the output vector C
    tl.store(C_ptr + row_start * stride_C_n, C, mask=tl.arange(0, BLOCK_N) < N)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 32}, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 32}, num_warps=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 32}, num_warps=1),
    ],
    key=['N', 'M']
)
def mv(A, B, C, N, M):
    # Assert dimension compatibility
    assert A.shape[1] == B.shape[0]
    assert C.shape[0] == A.shape[0]

    # Compute grid dimensions
    grid = (triton.cdiv(N, BLOCK_N),)

    # Launch the kernel
    mv_kernel[grid](
        A, B, C, N, M,
        A.stride(0), A.stride(1),
        C.stride(0),
        BLOCK_N, BLOCK_M
    )
