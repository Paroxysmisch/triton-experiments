import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 32}, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 128}, num_warps=4),
    ],
    key=['N', 'M']
)
@triton.jit
def mv_kernel(
    A_ptr,  # Pointer to the matrix A
    B_ptr,  # Pointer to the vector B
    C_ptr,  # Pointer to the output vector C
    N,      # Number of rows in A
    M,      # Number of columns in A (and length of B)
    stride_A_N,  # Stride of A along the N dimension
    stride_A_M,  # Stride of A along the M dimension
    stride_B,    # Stride of B (which is 1)
    stride_C,    # Stride of C (which is 1)
    BLOCK_N: tl.constexpr,  # Block size for N
    BLOCK_M: tl.constexpr   # Block size for M
):
    # Compute the row index for this block
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_N

    # Compute the column index for this block
    col_start = tl.program_id(axis=1) * BLOCK_M

    # Initialize the output block
    C_block = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Iterate over the columns of A and elements of B
    for col in range(col_start, col_start + BLOCK_M):
        if col < M:
            # Load the column of A and the element of B
            A_block = tl.load(A_ptr + row_start * stride_A_N + col * stride_A_M, mask=row_start + tl.arange(0, BLOCK_N) < N, other=0.0)
            B_val = tl.load(B_ptr + col * stride_B)
            # Perform the dot product
            C_block += A_block * B_val

    # Store the result in C
    tl.store(C_ptr + row_start * stride_C, C_block, mask=row_start + tl.arange(0, BLOCK_N) < N)

import torch
import triton

# Define the matrix-vector multiplication function
def matrix_vector_multiply(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
    assert A.dim() == 2, "A must be a 2D tensor"
    assert B.dim() == 1, "B must be a 1D tensor"
    assert C.dim() == 1, "C must be a 1D tensor"
    assert A.shape[1] == B.shape[0], "A and B dimensions must be compatible"
    assert A.shape[0] == C.shape[0], "A and C dimensions must be compatible"

    N, M = A.shape

    # Launch the kernel
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']), triton.cdiv(M, meta['BLOCK_M']))
    mv_kernel[grid](
        A, B, C,
        N, M,
        A.stride(0), A.stride(1),
        B.stride(0),
        C.stride(0)
    )

# Example usage
N, M = 1024, 512
A = torch.randn((N, M), device='cuda')
B = torch.randn((M,), device='cuda')
C = torch.zeros((N,), device='cuda')

matrix_vector_multiply(A, B, C)

# Verify the result
C_ref = torch.matmul(A, B)
print("Max error:", torch.max(torch.abs(C - C_ref)))
