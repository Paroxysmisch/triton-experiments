triton
import triton
import triton.language as tl

@triton.jit
def matmul_kernel_persistent(
    a_ptr: tl.tensor, b_ptr: tl.tensor, c_ptr: tl.tensor,
    M: tl.int32, N: tl.int32, K: tl.int32,
    stride_a: tl.int32, stride_b: tl.int32, stride_c: tl.int32,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Thread indices
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    pid_x = pid % tl.cdiv(N, BLOCK_SIZE_N)
    pid_y = pid // tl.cdiv(N, BLOCK_SIZE_N)
    
    # Block indices
    bid_m = pid_y * BLOCK_SIZE_N + pid_x
    bid_k = pid_x
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over tiles of K
    for k in range(0, K, BLOCK_SIZE_K):
        # Load a tile of A and B into shared memory
        a = tl.load(a_ptr + bid_m * stride_a + k * stride_a, mask=k + BLOCK_SIZE_K < K, other=0.0)
        b = tl.load(b_ptr + k * stride_b + bid_k * stride_b, mask=k + BLOCK_SIZE_K < K, other=0.0)
        
        # Perform matrix multiplication for the tile
        acc += a[:, None] * b[None, :]
    
    # Write back the result to global memory
    c_ptr[bid_m * stride_c + pid_x * stride_c] = acc
