import triton
import triton.language as tl

@triton.jit
def solve_kernel(A_ptr, B_ptr, A_inv_ptr, B_out_ptr, N, dtype: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)
    x = pid % BLOCK_SIZE
    y = pid // BLOCK_SIZE
    
    A_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=dtype)
    B_block = tl.zeros((BLOCK_SIZE, 1), dtype=dtype)
    
    A_inv_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=dtype)
    B_out_block = tl.zeros((BLOCK_SIZE, 1), dtype=dtype)
    
    A_load_ptr = A_ptr + y * N * BLOCK_SIZE + x * BLOCK_SIZE
    B_load_ptr = B_ptr + y * N * BLOCK_SIZE + x
    
    A_store_ptr = A_inv_ptr + y * N * BLOCK_SIZE + x * BLOCK_SIZE
    B_store_ptr = B_out_ptr + y * N * BLOCK_SIZE + x
    
    for i in range(BLOCK_SIZE):
        A_block[i] = tl.load(A_load_ptr + i * N, mask=i < N, other=0)
        B_block[i] = tl.load(B_load_ptr + i, mask=i < N, other=0)
    
    # Compute the inverse of A_block
    det = A_block[0][0] * A_block[1][1] - A_block[0][1] * A_block[1][0]
    inv_det = 1.0 / det
    
    A_inv_block[0][0] = A_block[1][1] * inv_det
    A_inv_block[0][1] = -A_block[0][1] * inv_det
    A_inv_block[1][0] = -A_block[1][0] * inv_det
    A_inv_block[1][1] = A_block[0][0] * inv_det
    
    # Perform matrix multiplication
    for j in range(BLOCK_SIZE):
        B_out_block[j] = A_inv_block[0][j] * B_block[0] + A_inv_block[1][j] * B_block[1]
    
    tl.store(A_store_ptr, A_inv_block, mask=True)
    tl.store(B_store_ptr, B_out_block, mask=True)

# Define the block size
BLOCK_SIZE = 32
