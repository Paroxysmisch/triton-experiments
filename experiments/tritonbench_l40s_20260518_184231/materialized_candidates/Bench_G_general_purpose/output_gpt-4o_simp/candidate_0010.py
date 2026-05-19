import triton
import triton.language as tl

# Triton kernel for matrix-vector multiplication
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 32}),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 128}),
    ],
    key=['N', 'M']
)
@triton.jit
def mv_kernel(A_ptr, B_ptr, C_ptr, N, M, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    # Define the block index
    pid = tl.program_id(axis=0)
    
    # Define the starting indices for this block
    block_n_start = pid * BLOCK_N
    
    # Create a pointer to the start of the block in C
    C_block_ptr = C_ptr + block_n_start
    
    # Create an accumulator for the results
    C_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    # Iterate over the blocks of the matrix
    for block_m_start in range(0, M, BLOCK_M):
        # Load the sub-matrix A and sub-vector B
        A_block_ptr = A_ptr + block_n_start * M + block_m_start
        B_block_ptr = B_ptr + block_m_start
        
        A_block = tl.load(A_block_ptr + tl.arange(0, BLOCK_N)[:, None] * M + tl.arange(0, BLOCK_M)[None, :], mask=True)
        B_block = tl.load(B_block_ptr + tl.arange(0, BLOCK_M), mask=True)
        
        # Perform the matrix-vector multiplication for this block
        C_acc += tl.dot(A_block, B_block)
    
    # Store the result in C
    tl.store(C_block_ptr + tl.arange(0, BLOCK_N), C_acc)

# Python wrapper to call the Triton kernel
def matrix_vector_multiply(A, B, N, M):
    # Ensure input dimensions are correct
    assert A.shape == (N, M)
    assert B.shape == (M,)
    
    # Allocate output vector C
    C = torch.empty((N,), dtype=A.dtype, device=A.device)
    
    # Launch the Triton kernel
    grid = (triton.cdiv(N, 64),)  # Assuming BLOCK_N=64 for the grid size
    mv_kernel[grid](A, B, C, N, M)
    
    return C

# Example usage
import torch

# Define matrix A and vector B
N, M = 128, 256
A = torch.randn((N, M), dtype=torch.float32, device='cuda')
B = torch.randn((M,), dtype=torch.float32, device='cuda')

# Perform matrix-vector multiplication
C = matrix_vector_multiply(A, B, N, M)
print(C)
