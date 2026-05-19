import triton
import triton.language as tl

# Define the matrix multiplication kernel
@triton.jit
def matmul_kernel(
    A, B, C, 
    stride_a_row, stride_a_col, 
    stride_b_row, stride_b_col, 
    stride_c_row, stride_c_col, 
    M, N, K, 
    use_leaky_relu: tl.constexpr, 
    alpha: tl.float32, 
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr
):
    # Compute the program ID in a 2D grid
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_warp = num_pid_m * num_pid_n
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Compute the block offsets
    rm = pid_m * BLOCK_SIZE_M
    rn = pid_n * BLOCK_SIZE_N

    # Compute the block bounds
    rm_bound = tl.minimum(rm + BLOCK_SIZE_M, M)
    rn_bound = tl.minimum(rn + BLOCK_SIZE_N, N)

    # Initialize the accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load the A and B blocks
        a_tile = tl.load(A + rm * stride_a_row + k * stride_a_col, mask=rm < M and k < K, other=0.0)
        b_tile = tl.load(B + k * stride_b_row + rn * stride_b_col, mask=k < K and rn < N, other=0.0)

        # Perform the matrix multiplication
        accumulator += tl.dot(a_tile, b_tile)

    # Apply the leaky ReLU activation if requested
    if use_leaky_relu:
        accumulator = tl.where(accumulator > 0, accumulator, accumulator * alpha)

    # Store the result in the output matrix C
    tl.store(C + rm * stride_c_row + rn * stride_c_col, accumulator, mask=rm < M and rn < N)

# Define the wrapper function for the kernel
def matmul(A, B, C, use_leaky_relu=False, alpha=0.01, BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16):
    # Validate the input shapes
    M, K = A.shape
    K, N = B.shape
    assert C.shape == (M, N), "Output matrix C must have shape (M, N)"
    assert A.dtype == B.dtype == C.dtype, "Input matrices A, B, and C must have the same data type"
    assert A.dtype in [tl.float32, tl.float16], "Input matrices must be of type float32 or float16"

    # Set up the grid for execution
    grid = (tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N),)

    # Launch the kernel
    matmul_kernel[grid](
        A, B, C, 
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1), 
        M, N, K, 
        use_leaky_relu, 
        alpha, 
        BLOCK_SIZE_M, 
        BLOCK_SIZE_N, 
        BLOCK_SIZE_K
    )
