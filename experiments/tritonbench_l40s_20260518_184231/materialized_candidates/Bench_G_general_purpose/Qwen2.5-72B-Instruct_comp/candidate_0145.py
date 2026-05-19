import triton
import triton.language as tl

@triton.jit
def dequantize_kernel(
    b_ptr,  # Pointer to the input int8 matrix
    b_scale_ptr,  # Pointer to the scale factors
    fpb_ptr,  # Pointer to the output float matrix
    K: tl.constexpr,  # Number of columns in the input matrix
    N: tl.constexpr,  # Number of rows in the input matrix
    b_stride_row: tl.constexpr,  # Stride for rows in the input matrix
    b_stride_col: tl.constexpr,  # Stride for columns in the input matrix
    b_scale_stride: tl.constexpr,  # Stride for scale factors
    fpb_stride_row: tl.constexpr,  # Stride for rows in the output matrix
    fpb_stride_col: tl.constexpr,  # Stride for columns in the output matrix
    BLOCK_SIZE_N: tl.constexpr,  # Block size for rows
    BLOCK_SIZE_K: tl.constexpr  # Block size for columns
):
    pid_n = tl.program_id(axis=0)  # Block index for rows
    pid_k = tl.program_id(axis=1)  # Block index for columns

    # Compute the block start indices
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    # Load the scale factor for the current block
    scale = tl.load(b_scale_ptr + pid_k * b_scale_stride)

    # Initialize the output block
    fpb_block = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    # Iterate over the block
    for n in range(BLOCK_SIZE_N):
        for k in range(BLOCK_SIZE_K):
            # Compute the global indices
            global_n = block_start_n + n
            global_k = block_start_k + k

            # Check if the indices are within bounds
            if global_n < N and global_k < K:
                # Load the int8 value
                b_val = tl.load(b_ptr + global_n * b_stride_row + global_k * b_stride_col)
                # Dequantize the value
                fpb_block[n, k] = b_val * scale

    # Store the dequantized block
    for n in range(BLOCK_SIZE_N):
        for k in range(BLOCK_SIZE_K):
            # Compute the global indices
            global_n = block_start_n + n
            global_k = block_start_k + k

            # Check if the indices are within bounds
            if global_n < N and global_k < K:
                # Store the dequantized value
                tl.store(fpb_ptr + global_n * fpb_stride_row + global_k * fpb_stride_col, fpb_block[n, k])

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256}, num_warps=4),
    ],
    key=['N', 'K']
)
@triton.jit
def dequantize_kernel(
    b_ptr,  # Pointer to the input int8 matrix
    b_scale_ptr,  # Pointer to the scale factors
    fpb_ptr,  # Pointer to the output float matrix
    K: tl.constexpr,  # Number of columns in the input matrix
    N: tl.constexpr,  # Number of rows in the input matrix
    b_stride_row: tl.constexpr,  # Stride for rows in the input matrix
    b_stride_col: tl.constexpr,  # Stride for columns in the input matrix
    b_scale_stride: tl.constexpr,  # Stride for scale factors
    fpb_stride_row: tl.constexpr,  # Stride for rows in the output matrix
    fpb_stride_col: tl.constexpr,  # Stride for columns in the output matrix
    BLOCK_SIZE_N: tl.constexpr,  # Block size for rows
    BLOCK_SIZE_K: tl.constexpr  # Block size for columns
):
    pid_n = tl.program_id(axis=0)  # Block index for rows
    pid_k = tl.program_id(axis=1)  # Block index for columns

    # Compute the block start indices
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_k = pid_k * BLOCK_SIZE_K

    # Load the scale factor for the current block
    scale = tl.load(b_scale_ptr + pid_k * b_scale_stride)

    # Initialize the output block
    fpb_block = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    # Iterate over the block
    for n in range(BLOCK_SIZE_N):
        for k in range(BLOCK_SIZE_K):
            # Compute the global indices
            global_n = block_start_n + n
            global_k = block_start_k + k

            # Check if the indices are within bounds
            if global_n < N and global_k < K:
                # Load the int8 value
                b_val = tl.load(b_ptr + global_n * b_stride_row + global_k * b_stride_col)
                # Dequantize the value
                fpb_block[n, k] = b_val * scale

    # Store the dequantized block
    for n in range(BLOCK_SIZE_N):
        for k in range(BLOCK_SIZE_K):
            # Compute the global indices
            global_n = block_start_n + n
            global_k = block_start_k + k

            # Check if the indices are within bounds
            if global_n < N and global_k < K:
                # Store the dequantized value
                tl.store(fpb_ptr + global_n * fpb_stride_row + global_k * fpb_stride_col, fpb_block[n, k])

def matmul_dequantize_int8(a: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor):
    # Ensure the input matrices have compatible dimensions
    assert a.shape[1] == b.shape[0], "Matrix dimensions must be compatible for multiplication"
    assert b_scale.shape[0] == b.shape[1], "Scale factors must match the number of columns in b"

    M, K = a.shape
    N = b.shape[0]

    # Allocate the output matrix
    fpb = torch.empty((N, K), dtype=torch.float32, device=a.device)

    # Launch the dequantize kernel
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']), triton.cdiv(K, META['BLOCK_SIZE_K']))
    dequantize_kernel[grid](
        b, b_scale, fpb,
        K, N,
        b.stride(0), b.stride(1),
        b_scale.stride(0),
        fpb.stride(0), fpb.stride(1)
    )

    # Perform the matrix multiplication
    c = torch.mm(a, fpb)

    return c
