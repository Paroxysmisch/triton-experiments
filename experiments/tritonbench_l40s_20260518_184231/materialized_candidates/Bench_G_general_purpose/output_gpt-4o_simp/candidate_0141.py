import triton
import triton.language as tl

@triton.jit
def matmul_kernel_persistent(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, dtype: tl.constexpr):
    pid = tl.program_id(0)
    
    # Compute the starting indices of the block
    block_row = pid // (N // BLOCK_SIZE_N)
    block_col = pid % (N // BLOCK_SIZE_N)
    
    # Create pointers for a, b, c
    a_block_ptr = a_ptr + block_row * BLOCK_SIZE_M * stride_am
    b_block_ptr = b_ptr + block_col * BLOCK_SIZE_N * stride_bn
    c_block_ptr = c_ptr + block_row * BLOCK_SIZE_M * stride_cm + block_col * BLOCK_SIZE_N * stride_cn
    
    # Create accumulators for the result
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=dtype)
    
    # Loop over the K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B
        a_block = tl.load(a_block_ptr + k * stride_ak, mask=(k + tl.arange(0, BLOCK_SIZE_K) < K))
        b_block = tl.load(b_block_ptr + k * stride_bk, mask=(k + tl.arange(0, BLOCK_SIZE_K) < K))
        
        # Compute matrix multiplication for the block
        acc += tl.dot(a_block, b_block)
    
    # Store the result back to C
    tl.store(c_block_ptr, acc)


def matmul_persistent(a, b, M, N, K, dtype):
    import triton

    # Determine block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Check input dimensions
    assert a.shape == (M, K), "Matrix A dimensions are incorrect"
    assert b.shape == (K, N), "Matrix B dimensions are incorrect"

    # Allocate output matrix
    c = triton.zeros((M, N), dtype=dtype)

    # Determine grid size
    grid = (M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N)

    # Launch the kernel
    matmul_kernel_persistent[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        dtype=dtype
    )

    return c
