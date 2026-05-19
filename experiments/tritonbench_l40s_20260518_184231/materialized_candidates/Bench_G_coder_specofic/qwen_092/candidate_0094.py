import triton
import triton.language as tl

# Define the block sizes
BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 16
BLOCK_SIZE_K = 8

@triton.jit
def matmul_kernel(
    a_ptr: tl.tensor, b_ptr: tl.tensor, c_ptr: tl.tensor,
    M: tl.int32, N: tl.int32, K: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Calculate the thread block's position in the grid
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    # Calculate the starting position of the sub-matrix in the output matrix
    row_base = pid_m * BLOCK_SIZE_M
    col_base = pid_n * BLOCK_SIZE_N

    # Initialize the accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Calculate offsets for loading sub-blocks of a and b
        a_offs = (row_base * stride_am) + (k * stride_ak)
        b_offs = (k * stride_bk) + (col_base * stride_bn)

        # Load sub-blocks of a and b
        a = tl.load(a_ptr + a_offs, (BLOCK_SIZE_M, BLOCK_SIZE_K), mask=pid_m < M and pid_k < K, boundary_check=(0, 0))
        b = tl.load(b_ptr + b_offs, (BLOCK_SIZE_K, BLOCK_SIZE_N), mask=pid_k < K and pid_n < N, boundary_check=(0, 0))

        # Perform dot product and accumulate
        accumulator += tl.dot(a, b)

    # Cast the result to float16 and store in c
    c_offs = (row_base * stride_cm) + (col_base * stride_cn)
    tl.store(c_ptr + c_offs, accumulator.to(tl.float16), mask=pid_m < M and pid_n < N)
