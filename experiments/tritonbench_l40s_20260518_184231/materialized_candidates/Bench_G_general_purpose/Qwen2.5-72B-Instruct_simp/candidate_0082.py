import triton
import triton.language as tl

@triton.jit
def load_reduce_kernel(
    x_ptr,  # Pointer to the input matrix
    y_ptr,  # Pointer to the output vector
    M,      # Number of rows in the input matrix
    N,      # Number of columns in the input matrix
    stride_xm,  # Stride for rows in the input matrix
    stride_xn,  # Stride for columns in the input matrix
    BLOCK_M: tl.constexpr,  # Block size for rows
    BLOCK_N: tl.constexpr   # Block size for columns
):
    # Compute the block ID in the M dimension
    pid_m = tl.program_id(0)
    # Compute the block ID in the N dimension
    pid_n = tl.program_id(1)

    # Compute the block start indices in the M and N dimensions
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N

    # Compute the block end indices in the M and N dimensions
    rm_end = tl.minimum(rm + BLOCK_M, M)
    rn_end = tl.minimum(rn + BLOCK_N, N)

    # Load the block of data from the input matrix
    x_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(rm, rm_end):
        for j in range(rn, rn_end):
            x_block[i - rm, j - rn] = tl.load(x_ptr + i * stride_xm + j * stride_xn)

    # Compute the maximum value along the second dimension
    max_values = tl.max(x_block, axis=1)

    # Write the result to the output vector
    for i in range(rm, rm_end):
        tl.store(y_ptr + i, max_values[i - rm])

import numpy as np
import torch

def load_reduce(M, N, BLOCK_M, BLOCK_N):
    # Generate random input data
    x = np.random.randn(M, N).astype(np.float32)
    x_ptr = triton.testing.to_triton(x, device='cuda')

    # Allocate output vector
    y = np.zeros(M, dtype=np.float32)
    y_ptr = triton.testing.to_triton(y, device='cuda')

    # Define grid and block dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), 1)

    # Launch the kernel
    load_reduce_kernel[grid](
        x_ptr, y_ptr, M, N, x.shape[1], 1, BLOCK_M, BLOCK_N
    )

    # Copy the result back to the host
    y = y_ptr.to_numpy()

    # Validate the result
    expected = np.max(x, axis=1)
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    print("Test passed!")

# Example usage
M, N = 1024, 1024
BLOCK_M, BLOCK_N = 32, 32
load_reduce(M, N, BLOCK_M, BLOCK_N)
