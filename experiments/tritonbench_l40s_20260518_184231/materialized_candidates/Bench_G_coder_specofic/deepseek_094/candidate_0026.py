import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_A=None, stride_B=None, stride_C=None,
    BLOCK_SIZE=128):

    # Define the grid
    grid = lambda M, N: ((M + BLOCK_SIZE - 1) // BLOCK_SIZE, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    n_warps = lambda M, N: grid(M, N)[0] * grid(M, N)[1]

    # Define the program ids
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Define the block ids
    bid_x = tl.block_id(axis=0)
    bid_y = tl.block_id(axis=1)

    # Define the warp id
    warp_id = pid_x * grid(M, N)[1] + pid_y

    # Define the warp size
    WARP_SIZE = 32

    # Define the lane id
    lane_id = tl.arange(0, WARP_SIZE)

    # Define the memory pointers
    A_ptrs = A_ptr + bid_x * BLOCK_SIZE * stride_A + lane_id
    B_ptrs = B_ptr + bid_y * BLOCK_SIZE * stride_B + lane_id
    C_ptrs = C_ptr + bid_x * BLOCK_SIZE * stride_C + lane_id

    # Define the matrix elements
    A = tl.load(A_ptrs, mask=lane_id < K)
    B = tl.load(B_ptrs, mask=lane_id < N)
    C = tl.load(C_ptrs, mask=lane_id < N)

    # Perform the matrix multiplication
    C += A * B

    # Store the matrix elements
    tl.store(C_ptrs, C, mask=lane_id < N)

def matmul(A, B, C):
    # Get the dimensions
    M, K = A.shape
    _, N = B.shape

    # Get the strides
    stride_A = tl.arange(0, M * K, step=K)
    stride_B = tl.arange(0, K * N, step=N)
    stride_C = tl.arange(0, M * N, step=N)

    # Get the pointers
    A_ptr = tl.to_ptr(A)
    B_ptr = tl.to_ptr(B)
    C_ptr = tl.to_ptr(C)

    # Call the kernel
    matmul_kernel[grid=(M, N), dynamic_shared=True](
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_A, stride_B, stride_C)
