import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_stages=3, num_warps=2),
    ],
    key=['N', 'M']
)
@triton.jit
def mv_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    # Compute the start of the block
    row_start = pid * BLOCK_N
    col_start = tl.arange(0, BLOCK_M)
    
    # Initialize the accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    # Load the vector B
    B = tl.load(B_ptr + col_start)
    
    # Loop over the columns of A
    for k in range(0, M, BLOCK_M):
        # Load a block of A
        A = tl.load(A_ptr + (row_start[:, None] + k) * M + col_start[None, :])
        # Compute partial dot product
        acc += tl.dot(A, B)
    
    # Write back the result
    tl.store(C_ptr + row_start, acc)

def matrix_vector_multiply(A, B):
    N, M = A.shape
    C = torch.empty(N, dtype=A.dtype, device=A.device)
    
    # Launch the kernel
    grid = lambda META: (triton.cdiv(N, META['BLOCK_N']),)
    mv_kernel[grid](A, B, C, N, M)
    
    return C

# Example usage
A = torch.randn(1024, 512, device='cuda')
B = torch.randn(512, device='cuda')
C = matrix_vector_multiply(A, B)
