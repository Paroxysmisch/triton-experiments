import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    x_ptr,  # Pointer to input matrix x
    y_ptr,  # Pointer to input matrix y
    z_ptr,  # Pointer to output matrix z
    M,      # Number of rows in matrix x
    N,      # Number of columns in matrix y
    K,      # Number of columns in matrix x (and rows in matrix y)
    stride_xm,  # Stride for rows in matrix x
    stride_xk,  # Stride for columns in matrix x
    stride_yn,  # Stride for columns in matrix y
    stride_yk,  # Stride for rows in matrix y
    stride_zm,  # Stride for rows in matrix z
    stride_zn,  # Stride for columns in matrix z
    BLOCK_SIZE_M: tl.constexpr,  # Block size for rows in matrix x
    BLOCK_SIZE_N: tl.constexpr,  # Block size for columns in matrix y
    BLOCK_SIZE_K: tl.constexpr,  # Block size for columns in matrix x (and rows in matrix y)
    GROUP_SIZE_M: tl.constexpr   # Group size for rows in matrix x
):
    # Compute the program id for the current block
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute the offsets for the current block
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) * stride_xm
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) * stride_yn
    offs_k = tl.arange(0, BLOCK_SIZE_K) * stride_xk

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load the sub-matrices of x and y
        a = tl.load(x_ptr + offs_am[:, None] + offs_k[None, :], mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(y_ptr + offs_k[:, None] + offs_bn[None, :], mask=offs_k[:, None] < K - k, other=0.0)

        # Compute the product of the sub-matrices
        acc += tl.dot(a, b)

    # Compute the offsets for the output matrix z
    offs_zm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_zn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    z_offsets = offs_zm[:, None] * stride_zm + offs_zn[None, :] * stride_zn

    # Store the result in global memory
    tl.store(z_ptr + z_offsets, acc)

import torch

def matmul(x, y):
    # Get the dimensions of the input matrices
    M, K = x.shape
    K, N = y.shape

    # Allocate the output matrix
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Define the block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 8

    # Calculate the grid size
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Launch the kernel
    matmul_kernel[grid](
        x, y, z,
        M, N, K,
        x.stride(0), x.stride(1),
        y.stride(1), y.stride(0),
        z.stride(0), z.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )

    return z
