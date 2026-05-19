import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def matmul_kernel_persistent(a_ptr, b_ptr, c_ptr, M, N, K, stride_a_m, stride_a_k, stride_b_k, stride_b_n, stride_c_m, stride_c_n, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, NUM_SMS: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Calculate block indices
    block_m = pid // (N // BLOCK_SIZE_N)
    block_n = pid % (N // BLOCK_SIZE_N)
    
    # Initialize accumulators
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension in BLOCK_SIZE_K tiles
    for k in range(0, K, BLOCK_SIZE_K):
        # Load a block of A and B
        a_block = tl.load(a_ptr + (block_m * BLOCK_SIZE_M * stride_a_m + k * stride_a_k) + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_a_m + tl.arange(0, BLOCK_SIZE_K), mask=(k + tl.arange(0, BLOCK_SIZE_K) < K))
        b_block = tl.load(b_ptr + (k * stride_b_k + block_n * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_b_k + tl.arange(0, BLOCK_SIZE_N), mask=(k + tl.arange(0, BLOCK_SIZE_K) < K))
        
        # Accumulate the product
        accumulator += tl.dot(a_block, b_block)
    
    # Store the result
    c_block = tl.load(c_ptr + (block_m * BLOCK_SIZE_M * stride_c_m + block_n * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_c_m + tl.arange(0, BLOCK_SIZE_N), mask=(block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < M) & (block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) < N))
    c_block += accumulator.to(c_block.dtype)
    tl.store(c_ptr + (block_m * BLOCK_SIZE_M * stride_c_m + block_n * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_c_m + tl.arange(0, BLOCK_SIZE_N), c_block, mask=(block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) < M) & (block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) < N))

# Define the wrapper function
def matmul_persistent(a, b):
    # Validate input dimensions and types
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
    assert a.dtype == b.dtype, "Matrices must have the same data type"
    
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output tensor
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    
    # Define block sizes and other kernel parameters
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    NUM_SMS = 80  # Example number, adjust based on your GPU
    
    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    matmul_kernel_persistent[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        M=M, N=N, K=K,
        stride_a_m=a.stride(0), stride_a_k=a.stride(1),
        stride_b_k=b.stride(0), stride_b_n=b.stride(1),
        stride_c_m=c.stride(0), stride_c_n=c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, NUM_SMS=NUM_SMS
    )
    
    return c

# Example usage
a = torch.randn(1024, 512, device='cuda', dtype=torch.float32)
b = torch.randn(512, 1024, device='cuda', dtype=torch.float32)
c = matmul_persistent(a, b)
