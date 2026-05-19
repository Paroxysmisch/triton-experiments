import triton
import triton.language as tl

@triton.jit
def bmm_kernel(X_ptr, Y_ptr, Z_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    i = pid // (BLOCK_SIZE_N * BLOCK_SIZE_K)
    j = (pid % (BLOCK_SIZE_N * BLOCK_SIZE_K)) // BLOCK_SIZE_K
    k = pid % BLOCK_SIZE_K
    
    x = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    y = tl.load(Y_ptr + ((j * K + k) * M))
    
    for m in range(BLOCK_SIZE_M):
        x[m] += tl.dot(x[m], y[m])
    
    z_idx = (i * N * M) + (j * M) + m
    tl.store(Z_ptr + z_idx, x)
