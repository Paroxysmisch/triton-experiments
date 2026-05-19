import triton
import triton.language as tl

@triton.jit
def batched_vecmat_kernel(
    A_ptr,  # Pointer to the input matrix A (batch, M, K)
    B_ptr,  # Pointer to the input matrix B (batch, K, N)
    C_ptr,  # Pointer to the output matrix C (batch, M, N)
    M,      # Number of rows in A and C
    N,      # Number of columns in B and C
    K,      # Number of columns in A and rows in B
    stride_A_batch,  # Stride of A in the batch dimension
    stride_A_m,      # Stride of A in the M dimension
    stride_A_k,      # Stride of A in the K dimension
    stride_B_batch,  # Stride of B in the batch dimension
    stride_B_k,      # Stride of B in the K dimension
    stride_B_n,      # Stride of B in the N dimension
    stride_C_batch,  # Stride of C in the batch dimension
    stride_C_m,      # Stride of C in the M dimension
    stride_C_n,      # Stride of C in the N dimension
    BLOCK_SIZE_M: tl.constexpr,  # Block size for the M dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for the N dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for the K dimension
    GROUP_SIZE_M: tl.constexpr    # Group size for the M dimension
):
    # Compute the batch index
    batch_idx = tl.program_id(2)
    # Compute the block indices in the M and N dimensions
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = GROUP_SIZE_M * group_id
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute the block offsets in the M and N dimensions
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Compute the batch offsets
    A_batch_ptr = A_ptr + batch_idx * stride_A_batch
    B_batch_ptr = B_ptr + batch_idx * stride_B_batch
    C_batch_ptr = C_ptr + batch_idx * stride_C_batch

    # Initialize the output tile
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load the input tiles
        a = tl.load(A_batch_ptr + offs_m[:, None] * stride_A_m + (k + offs_k[None, :]) * stride_A_k, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(B_batch_ptr + (k + offs_k[:, None]) * stride_B_k + offs_n[None, :] * stride_B_n, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Perform the matrix multiplication
        acc += tl.dot(a, b)

    # Store the result in the output matrix
    C_batch_ptr += offs_m[:, None] * stride_C_m + offs_n[None, :] * stride_C_n
    tl.store(C_batch_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

import triton
import triton.language as tl
import torch

def batched_vecmat(A, B, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M):
    # Ensure A and B are on the same device
    assert A.device == B.device, "A and B must be on the same device"
    device = A.device

    # Ensure A and B have the correct dimensions
    assert A.shape[1] == M and A.shape[2] == K, f"A must have shape (batch, {M}, {K})"
    assert B.shape[1] == K and B.shape[2] == N, f"B must have shape (batch, {K}, {N})"

    # Allocate the output matrix C
    C = torch.empty((A.shape[0], M, N), device=device, dtype=A.dtype)

    # Define the grid and block dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
        A.shape[0]
    )

    # Launch the kernel
    batched_vecmat_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )

    return C

import torch

# Define the matrix dimensions
M, N, K = 1024, 1024, 1024
batch_size = 32

# Create random input matrices A and B
A = torch.randn((batch_size, M, K), device='cuda', dtype=torch.float32)
B = torch.randn((batch_size, K, N), device='cuda', dtype=torch.float32)

# Define the block sizes
BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 16
BLOCK_SIZE_K = 16
GROUP_SIZE_M = 8

# Perform the batched vector-matrix multiplication
C = batched_vecmat(A, B, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)

# Print the result
print(C)
