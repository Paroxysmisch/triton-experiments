triton
import triton
import triton.language as tl

# Define the constants
BLOCK_N = 32
BLOCK_M = 16

@triton.jit
def mv_kernel(A, B, C, N, M):
    """
    Matrix-vector multiplication kernel.
    
    Parameters:
    A (float32): Input matrix of size N x M
    B (float32): Input vector of size M
    C (float32): Output vector of size N
    N (int32): Number of rows in matrix A
    M (int32): Number of columns in matrix A and size of vector B
    """
    # Define the thread indices
    pid = tl.program_id(axis=0)  # Index of the program in the grid
    num_programs = tl.cdiv(N, BLOCK_N)  # Total number of programs in the grid

    # Define the block indices
    row = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    col = tl.arange(0, BLOCK_M)

    # Initialize the output for this thread
    output = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Load sub-matrices and vector blocks into shared memory
    A_block = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    B_block = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for m in range(0, M, BLOCK_M):
        A_block = tl.load(A + row[:, None] * M + col)
        B_block = tl.load(B + m + col)
        output += A_block * B_block

    # Reduce along the rows and store in the output vector C
    C[pid * BLOCK_N : pid * BLOCK_N + BLOCK_N] = tl.reduce(output, axis=0, op=tl.sum)

@triton.jit
def mv(A, B, C, N, M):
    """
    Wrapper function for matrix-vector multiplication.
    
    Parameters:
    A (float32): Input matrix of size N x M
    B (float32): Input vector of size M
    C (float32): Output vector of size N
    N (int32): Number of rows in matrix A
    M (int32): Number of columns in matrix A and size of vector B
    """
    # Assert dimension compatibility
    assert A.shape[0] == N and A.shape[1] == M, "Matrix A dimensions are incorrect"
    assert B.shape[0] == M, "Vector B dimensions are incorrect"
    assert C.shape[0] == N, "Output vector C dimensions are incorrect"

    # Calculate the grid dimensions
    grid = (triton.cdiv(N, BLOCK_N),)

    # Launch the kernel
    mv_kernel[grid](A, B, C, N, M)
