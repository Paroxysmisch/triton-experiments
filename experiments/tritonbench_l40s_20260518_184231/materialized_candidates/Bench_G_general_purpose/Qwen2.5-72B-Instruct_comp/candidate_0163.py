import triton
import triton.language as tl

# Define the leaky ReLU activation function
@triton.jit
def leaky_relu(x, negative_slope=0.01):
    return tl.where(x > 0, x, x * negative_slope)

# Define the matrix multiplication kernel
@triton.jit
def matmul_kernel(
    A, B, C,  # Pointers to matrices
    M, N, K,  # Matrix dimensions
    stride_am, stride_ak,  # Strides for matrix A
    stride_bk, stride_bn,  # Strides for matrix B
    stride_cm, stride_cn,  # Strides for matrix C
    BLOCK_SIZE_M: tl.constexpr,  # Block size for M dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for N dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for K dimension
    GROUP_SIZE_M: tl.constexpr,  # Group size for M dimension
    activation: tl.constexpr  # Activation function name
):
    # Compute the program ID in a 1D grid
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute the block bounds
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    A_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulate the result
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A_ptrs)
        b = tl.load(B_ptrs)
        accumulator += tl.dot(a, b)
        A_ptrs += BLOCK_SIZE_K * stride_ak
        B_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply the activation function if specified
    if activation == "leaky_relu":
        accumulator = leaky_relu(accumulator)

    # Store the result back to the output matrix C
    C_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_ptrs, accumulator)

import torch

def matmul(A, B, activation=None):
    # Get the dimensions of the input matrices
    M, K = A.shape
    K, N = B.shape

    # Initialize the output matrix C
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Define the block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    # Compute the grid size
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Launch the kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        activation
    )

    return C
