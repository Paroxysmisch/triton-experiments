import triton
import triton.language as tl

@triton.jit
def _swiglu_fwd_kernel(
    X,  # pointer to the input matrix X
    Y,  # pointer to the input matrix Y
    OUT,  # pointer to the output matrix
    M,  # number of rows in the matrices
    N,  # number of columns in the matrices
    stride_xm,  # stride of X along the row dimension
    stride_xn,  # stride of X along the column dimension
    stride_ym,  # stride of Y along the row dimension
    stride_yn,  # stride of Y along the column dimension
    stride_om,  # stride of OUT along the row dimension
    stride_on,  # stride of OUT along the column dimension
    BLOCK_SIZE: tl.constexpr,  # block size for parallelization
):
    # Compute the grid of blocks
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(M, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, M)

    # Iterate over the block
    for i in range(block_start, block_end):
        # Compute the sigmoid of X
        x = tl.load(X + i * stride_xm)
        sigmoid_x = 1 / (1 + tl.exp(-x))

        # Load Y
        y = tl.load(Y + i * stride_ym)

        # Compute the output
        out = sigmoid_x * y

        # Store the output
        tl.store(OUT + i * stride_om, out)

import torch
from typing import Optional

def _swiglu_fwd(X: torch.Tensor, Y: torch.Tensor, out: Optional[torch.Tensor] = None):
    # Ensure the input tensors are contiguous
    X = X.contiguous()
    Y = Y.contiguous()

    # Get the dimensions of the input tensors
    M, N = X.shape

    # Allocate the output tensor if not provided
    if out is None:
        out = torch.empty_like(X)

    # Define the grid and block dimensions
    BLOCK_SIZE = 256
    grid = (tl.cdiv(M, BLOCK_SIZE),)

    # Launch the kernel
    _swiglu_fwd_kernel[grid](
        X, Y, out,
        M, N,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE
    )

    return out
