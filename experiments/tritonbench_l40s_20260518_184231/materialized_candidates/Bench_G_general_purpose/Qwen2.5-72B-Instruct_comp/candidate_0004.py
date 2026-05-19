import triton
import triton.language as tl

@triton.jit
def _swiglu_fwd_kernel(
    X,  # Pointer to the input tensor X
    Y,  # Pointer to the input tensor Y
    OUT,  # Pointer to the output tensor
    M,  # Number of rows
    N,  # Number of columns
    stride_xm,  # Stride of X in the row direction
    stride_xn,  # Stride of X in the column direction
    stride_ym,  # Stride of Y in the row direction
    stride_yn,  # Stride of Y in the column direction
    stride_outm,  # Stride of OUT in the row direction
    stride_outn,  # Stride of OUT in the column direction
    BLOCK_SIZE: tl.constexpr,  # Block size for autotuning
):
    # Get the program ID in the row and column dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the block of columns this program instance will process
    block_start_n = pid_n * BLOCK_SIZE
    block_end_n = min(block_start_n + BLOCK_SIZE, N)

    # Compute the row this program instance will process
    row_start_m = pid_m * BLOCK_SIZE
    row_end_m = min(row_start_m + BLOCK_SIZE, M)

    # Initialize the output block with zeros
    out_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Iterate over the elements in the block
    for n in range(block_start_n, block_end_n):
        for m in range(row_start_m, row_end_m):
            # Load the elements from X and Y
            x = tl.load(X + m * stride_xm + n * stride_xn)
            y = tl.load(Y + m * stride_ym + n * stride_yn)

            # Compute the Swiglu operation: x * sigmoid(x) * y
            out = x * tl.sigmoid(x) * y

            # Store the result in the output block
            out_block[m - row_start_m, n - block_start_n] = out

    # Store the output block to the output tensor
    for n in range(block_start_n, block_end_n):
        for m in range(row_start_m, row_end_m):
            tl.store(OUT + m * stride_outm + n * stride_outn, out_block[m - row_start_m, n - block_start_n])

import torch
import triton
import triton.runtime

def _swiglu_fwd(xy: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensor is contiguous
    xy = xy.contiguous()

    # Split the input tensor into X and Y
    x, y = torch.split(xy, xy.shape[1] // 2, dim=1)

    # Get the dimensions
    M, N = x.shape

    # Initialize the output tensor
    out = torch.empty_like(x)

    # Define the grid configuration
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))

    # Launch the kernel
    _swiglu_fwd_kernel[grid](
        x, y, out,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=128
    )

    return out
