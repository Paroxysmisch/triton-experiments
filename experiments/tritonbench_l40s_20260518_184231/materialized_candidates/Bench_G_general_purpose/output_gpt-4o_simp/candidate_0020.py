import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # Define block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the start of the block in each dimension
    block_m_start = pid_m * BLOCK_SIZE_M
    block_n_start = pid_n * BLOCK_SIZE_N

    # Create a range for the block
    range_m = block_m_start + tl.arange(0, BLOCK_SIZE_M)
    range_n = block_n_start + tl.arange(0, BLOCK_SIZE_N)

    # Initialize the accumulator for the result block
    c_accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the K dimension in BLOCK_SIZE_K chunks
    for k in range(0, K, BLOCK_SIZE_K):
        # Create a range for the K dimension
        range_k = k + tl.arange(0, BLOCK_SIZE_K)

        # Load blocks of A and B
        a_block = tl.load(a_ptr + range_m[:, None] * stride_am + range_k[None, :] * stride_ak, mask=(range_m[:, None] < M) & (range_k[None, :] < K), other=0.0)
        b_block = tl.load(b_ptr + range_k[:, None] * stride_bk + range_n[None, :] * stride_bn, mask=(range_k[:, None] < K) & (range_n[None, :] < N), other=0.0)

        # Perform the matrix multiplication for this block
        c_accum += tl.dot(a_block, b_block)

    # Write the result to the output matrix C
    c_ptr = c_ptr + range_m[:, None] * stride_cm + range_n[None, :] * stride_cn
    tl.store(c_ptr, c_accum, mask=(range_m[:, None] < M) & (range_n[None, :] < N))

def matmul(a, b, c, M, N, K):
    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Calculate strides for the matrices
    stride_am = a.stride(0)
    stride_ak = a.stride(1)
    stride_bk = b.stride(0)
    stride_bn = b.stride(1)
    stride_cm = c.stride(0)
    stride_cn = c.stride(1)

    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    matmul_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        M=M, N=N, K=K,
        stride_am=stride_am, stride_ak=stride_ak,
        stride_bk=stride_bk, stride_bn=stride_bn,
        stride_cm=stride_cm, stride_cn=stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
