import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_a_m, stride_a_k, stride_b_k, stride_b_n, stride_c_m, stride_c_n, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M):
    # Program ID corresponds to the position of the block in the grid
    pid = tl.program_id(0)
    
    # Compute block row and column indices
    block_row = pid // (N // BLOCK_SIZE_N)
    block_col = pid % (N // BLOCK_SIZE_N)
    
    # Compute the start indices for each block
    start_m = block_row * BLOCK_SIZE_M
    start_n = block_col * BLOCK_SIZE_N
    
    # Create a pointer for each block of c
    c_ptrs = c_ptr + start_m * stride_c_m + start_n * stride_c_n
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks of a and b
        a_ptrs = a_ptr + start_m * stride_a_m + k * stride_a_k
        b_ptrs = b_ptr + k * stride_b_k + start_n * stride_b_n
        
        a = tl.load(a_ptrs, mask=(start_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K), other=0)
        b = tl.load(b_ptrs, mask=(k + tl.arange(0, BLOCK_SIZE_K)[:, None] < K) & (start_n + tl.arange(0, BLOCK_SIZE_N)[None, :] < N), other=0)
        
        # Perform matrix multiplication on the blocks
        acc += tl.dot(a, b)
    
    # Store the result
    c = acc.to(tl.int32)
    tl.store(c_ptrs, c, mask=(start_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & (start_n + tl.arange(0, BLOCK_SIZE_N)[None, :] < N))

def matmul(a, b):
    assert a.dtype == tl.int8 and b.dtype == tl.uint8, "Matrix 'a' must be int8 and matrix 'b' must be uint8"
    assert a.shape[1] == 4 * b.shape[0], "Incompatible dimensions for matrix multiplication"
    
    M, K4 = a.shape
    K = K4 // 4
    K, N = b.shape
    
    # Initialize output matrix c
    c = tl.zeros((M, N), dtype=tl.int32)
    
    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Strides
    stride_a_m, stride_a_k = a.stride(0), a.stride(1)
    stride_b_k, stride_b_n = b.stride(0), b.stride(1)
    stride_c_m, stride_c_n = c.stride(0), c.stride(1)
    
    # Define grid size
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # Launch the kernel
    matmul_kernel[grid](
        a, b, c, M, N, K,
        stride_a_m, stride_a_k, stride_b_k, stride_b_n, stride_c_m, stride_c_n,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )
    
    return c
