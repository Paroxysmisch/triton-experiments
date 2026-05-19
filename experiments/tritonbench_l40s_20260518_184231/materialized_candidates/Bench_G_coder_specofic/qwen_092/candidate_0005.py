import triton
import triton.language as tl

# Constants
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 32
GROUP_SIZE_M = 8

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a, stride_b, stride_c,
    BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M
):
    # Thread index within the block
    pid = tl.program_id(axis=0)
    pid_m = pid // (BLOCK_SIZE_N // GROUP_SIZE_M)
    pid_n = pid % (BLOCK_SIZE_N // GROUP_SIZE_M)
    pid_k = tl.program_id(axis=1)

    # Block index within the grid
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    grid_k = tl.cdiv(K, BLOCK_SIZE_K)

    # Indices for matrix a and b
    a_off = pid_m * BLOCK_SIZE_M * stride_a + pid_k * BLOCK_SIZE_K
    b_off = pid_k * BLOCK_SIZE_K * stride_b + pid_n * BLOCK_SIZE_N

    # Indices for matrix c
    c_off = pid_m * BLOCK_SIZE_M * stride_c + pid_n * BLOCK_SIZE_N

    # Initialize c to zero
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over k
    for k in range(0, K, BLOCK_SIZE_K):
        # Load a and b elements
        a = tl.load(a_ptr + a_off + k * stride_a, mask=k + BLOCK_SIZE_K < K, eviction_policy=tl.EvictionPolicy.LRU)
        b = tl.load(b_ptr + b_off + k * stride_b, mask=k + BLOCK_SIZE_K < K, eviction_policy=tl.EvictionPolicy.LRU)

        # Accumulate the dot product
        for i in range(0, BLOCK_SIZE_M, 4):
            for j in range(0, BLOCK_SIZE_N, 4):
                for kk in range(BLOCK_SIZE_K):
                    c[i:i+4, j:j+4] += a[i:i+4, kk] * b[kk, j:j+4]

    # Write back to global memory
    tl.store(c_ptr + c_off, c, mask=pid_m < grid_m and pid_n < grid_n)

@triton.jit
def matmul(
    a, b
):
    # Get the shapes of the input tensors
    M, K = a.shape
    K, N = b.shape

    # Check for dimensional compatibility
    assert K == b.shape[0], "Incompatible dimensions for matrix multiplication"

    # Initialize the output tensor
    c = tl.zeros((M, N), dtype=tl.int32)

    # Define the grid size
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    grid_k = tl.cdiv(K, BLOCK_SIZE_K)
    grid = (grid_m * grid_n * grid_k, 1)

    # Define the block size
    block_size = (BLOCK_SIZE_M // GROUP_SIZE_M * BLOCK_SIZE_N // GROUP_SIZE_M, BLOCK_SIZE_K)

    # Launch the kernel
    matmul_kernel[grid, block_size](a.data_ptr(), b.data_ptr(), c.data_ptr(), M, N, K, a.stride(0), b.stride(0), c.stride(0))

    return c
