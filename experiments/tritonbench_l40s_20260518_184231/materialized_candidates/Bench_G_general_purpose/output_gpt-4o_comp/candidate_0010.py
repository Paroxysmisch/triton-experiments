import triton
import triton.language as tl

@triton.jit
def mv_kernel(A_ptr, B_ptr, C_ptr, N, M, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    # Define the program IDs for each block
    pid_n = tl.program_id(0)
    
    # Compute the start and end of the block for this program
    block_start_n = pid_n * BLOCK_N
    block_end_n = min(block_start_n + BLOCK_N, N)
    
    # Initialize the accumulator for the result
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    # Iterate over the blocks of the matrix A
    for block_m in range(0, M, BLOCK_M):
        # Load a block of matrix A
        a_offset = block_start_n * M + block_m
        A_block = tl.load(A_ptr + a_offset, mask=(block_start_n < block_end_n)[:, None] & (block_m < M))
        
        # Load a block of vector B
        B_block = tl.load(B_ptr + block_m, mask=(block_m < M))
        
        # Perform the matrix-vector multiplication for the block
        acc += tl.dot(A_block, B_block)
    
    # Store the result in the output vector C
    c_offset = block_start_n
    tl.store(C_ptr + c_offset, acc, mask=(block_start_n < block_end_n))

def mv(A, B, C, BLOCK_N=128, BLOCK_M=128):
    # Assert dimension compatibility
    assert A.shape[1] == B.shape[0], "Incompatible dimensions for matrix-vector multiplication"
    N, M = A.shape
    
    # Launch the kernel with appropriate grid dimensions
    grid = (triton.cdiv(N, BLOCK_N),)
    
    # Execute the kernel
    mv_kernel[grid](A, B, C, N, M, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M)

# Example usage:
# Assuming A is a 2D tensor of shape (N, M) and B is a 1D tensor of shape (M,)
# C is a pre-allocated 1D tensor of shape (N,)
# mv(A, B, C)
