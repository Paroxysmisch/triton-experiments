import triton
import triton.language as tl

@triton.jit
def _quantize_global_transpose(
    A_ptr,  # Pointer to the input float matrix
    absmax_inv_ptr,  # Pointer to the inverse of the maximum absolute value
    B_ptr,  # Pointer to the output int8 matrix
    M,  # Number of rows in the input matrix
    N,  # Number of columns in the input matrix
    BLOCK_M: tl.constexpr,  # Block size for rows
    BLOCK_N: tl.constexpr  # Block size for columns
):
    # Compute the grid and block indices
    pid = tl.program_id(axis=0)
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    block_m = pid % num_blocks_m
    block_n = pid // num_blocks_m

    # Compute the block offsets
    rm = block_m * BLOCK_M
    rn = block_n * BLOCK_N

    # Compute the block bounds
    rm_bound = min(rm + BLOCK_M, M)
    rn_bound = min(rn + BLOCK_N, N)

    # Load the inverse of the maximum absolute value
    absmax_inv = tl.load(absmax_inv_ptr)

    # Quantize and transpose the block
    for m in range(rm, rm_bound):
        for n in range(rn, rn_bound):
            # Load the element from the input matrix
            a = tl.load(A_ptr + m * N + n)
            # Quantize the element
            q = tl.round(a * absmax_inv)
            # Clamp the quantized value to the int8 range
            q = tl.max(tl.min(q, 127), -128)
            # Store the transposed element in the output matrix
            tl.store(B_ptr + n * M + m, q)

### Python Wrapper Function

import torch

def quantize_global_transpose(A, absmax_inv):
    # Convert the input tensor to a contiguous format
    A = A.contiguous()
    M, N = A.shape

    # Allocate the output tensor
    B = torch.empty((N, M), dtype=torch.int8, device=A.device)

    # Define the grid and block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # Allocate the inverse of the maximum absolute value on the device
    absmax_inv_ptr = triton.make_ptr(absmax_inv, dtype=triton.float32)

    # Launch the Triton kernel
    _quantize_global_transpose[grid](
        A,  # Input matrix
        absmax_inv_ptr,  # Inverse of the maximum absolute value
        B,  # Output matrix
        M,  # Number of rows in the input matrix
        N,  # Number of columns in the input matrix
        BLOCK_M,  # Block size for rows
        BLOCK_N  # Block size for columns
    )

    return B
