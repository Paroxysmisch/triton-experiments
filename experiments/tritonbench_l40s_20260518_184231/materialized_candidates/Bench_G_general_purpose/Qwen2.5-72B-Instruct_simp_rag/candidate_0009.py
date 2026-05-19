import triton
import triton.language as tl
import torch

# Define the matrix-vector multiplication kernel
@triton.jit
def mv_kernel(
    A_ptr,  # Pointer to the matrix A
    B_ptr,  # Pointer to the vector B
    C_ptr,  # Pointer to the output vector C
    N,  # Number of rows in matrix A
    M,  # Number of columns in matrix A
    stride_A_N,  # Stride of matrix A in the N dimension
    stride_A_M,  # Stride of matrix A in the M dimension
    stride_B,  # Stride of vector B
    stride_C,  # Stride of vector C
    BLOCK_N: tl.constexpr,  # Block size in the N dimension
    BLOCK_M: tl.constexpr  # Block size in the M dimension
):
    # Compute the block ID in the N and M dimensions
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    # Compute the block bounds
    block_n_start = pid_n * BLOCK_N
    block_n_end = min(block_n_start + BLOCK_N, N)
    block_m_start = pid_m * BLOCK_M
    block_m_end = min(block_m_start + BLOCK_M, M)

    # Initialize the output block
    C_block = tl.zeros((BLOCK_N, 1), dtype=tl.float32)

    # Iterate over the block in the M dimension
    for m in range(block_m_start, block_m_end):
        # Load the vector B block
        B_block = tl.load(B_ptr + m * stride_B)

        # Iterate over the block in the N dimension
        for n in range(block_n_start, block_n_end):
            # Load the matrix A block
            A_block = tl.load(A_ptr + n * stride_A_N + m * stride_A_M)

            # Perform the multiplication and accumulate the result
            C_block[n - block_n_start, 0] += A_block * B_block

    # Store the result in the output vector C
    tl.store(C_ptr + block_n_start * stride_C, C_block)

def matrix_vector_multiply(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, BLOCK_N: int, BLOCK_M: int):
    # Ensure the tensors are on the same device
    assert A.device == B.device == C.device, "All tensors must be on the same device"
    assert A.dtype == B.dtype == C.dtype, "All tensors must have the same data type"
    assert A.shape[1] == B.shape[0], "Matrix A and vector B dimensions must be compatible"

    # Get the dimensions of the matrix and vector
    N, M = A.shape

    # Define the grid and block sizes
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M), 1)

    # Launch the kernel
    mv_kernel[grid](
        A, B, C,
        N, M,
        A.stride(0), A.stride(1),
        B.stride(0), C.stride(0),
        BLOCK_N, BLOCK_M
    )

# Example usage
N, M = 1024, 1024
A = torch.randn((N, M), device='cuda', dtype=torch.float32)
B = torch.randn((M,), device='cuda', dtype=torch.float32)
C = torch.zeros((N,), device='cuda', dtype=torch.float32)

# Define block sizes
BLOCK_N = 16
BLOCK_M = 16

# Perform the matrix-vector multiplication
matrix_vector_multiply(A, B, C, BLOCK_N, BLOCK_M)

# Print the result
print(C)
