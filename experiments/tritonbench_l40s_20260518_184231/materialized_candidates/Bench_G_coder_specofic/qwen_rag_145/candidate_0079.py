from typing import Tuple
import numpy as np
import triton.language as tl
from triton import jit

@jit
def _quantize_global_transpose(
    A: np.ndarray,
    B: np.ndarray,
    M: tl.constexpr, 
    N: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    GROUP_M: tl.constexpr,
    stride_am: tl.constexpr,
    stride_an: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bn: tl.constexpr,
    absmax_inv: tl.constexpr,
) -> None:
    # Program ID
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)

    # Create offsets for memory access
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load data from A
    a = tl.load(A + offsets_m[:, None] * stride_am + offsets_n[None, :] * stride_an)

    # Apply quantization
    b = a * absmax_inv * np.iinfo(np.int8).max

    # Store result in B
    tl.store(B + offsets_m[:, None] * stride_bm + offsets_n[None, :] * stride_bn, b)

def quantize_global_transpose(A, B, BLOCK_M, BLOCK_N, GROUP_M):
    # Calculate matrix dimensions
    M, N = A.shape[0], A.shape[1]

    # Calculate grid dimensions
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # Calculating absmax_inv
    absmax_inv = 1 / np.abs(A).max()

    # Define strides
    stride_an = N
    stride_am = stride_an * M
    stride_bn = N
    stride_bm = stride_bn * M

    # Call Triton kernel
    _quantize_global_transpose[(grid_m, grid_n, GROUP_M)](A, B, M, N, BLOCK_M, BLOCK_N, GROUP_M, stride_am, stride_an, stride_bm, stride_bn, absmax_inv)

# Call the function
A = np.random.rand(256, 256).astype(np.float32)
B = np.empty_like(A, dtype=np.int8)
quantize_global_transpose(A, B, BLOCK_M=128, BLOCK_N=128, GROUP_M=2)
