import triton
import triton.language as tl
import torch
import math

@triton.jit
def _matmul_update_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha, beta,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Compute a fused operation on blocks of C:
        C_block = alpha * (A_block * B_block) + beta * C_block
    M, N, K define the sizes for the GEMM (C is MxN, A is MxK, B is KxN).

    - A_ptr, B_ptr, C_ptr: pointers to A, B, C in memory
    - M, N, K: dimensions of the matrices
    - stride_am, stride_ak: strides for A
    - stride_bk, stride_bn: strides for B
    - stride_cm, stride_cn: strides for C
    - alpha, beta: scalars
    - BLOCK_M, BLOCK_N, BLOCK_K: tiling parameters
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Tile indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create pointers for C-tile
    # We'll load/update these in a fused manner
    C_tile_ptrs = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        # Compute K-slice
        k_range = tl.arange(0, BLOCK_K)
        k_index = k_start + k_range
        # Load A and B tiles
        A_tile_ptrs = A_ptr + (rm[:, None] * stride_am + k_index[None, :] * stride_ak)
        B_tile_ptrs = B_ptr + (k_index[:, None] * stride_bk + rn[None, :] * stride_bn)
        a = tl.load(A_tile_ptrs, mask=(rm[:, None] < M) & (k_index[None, :] < K), other=0.0)
        b = tl.load(B_tile_ptrs, mask=(k_index[:, None] < K) & (rn[None, :] < N), other=0.0)
        # Accumulate partial products
        acc += tl.dot(a, b)

    # Load old C, scale appropriately, and store result
    c_old = tl.load(C_tile_ptrs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    c_new = alpha * acc + beta * c_old
    tl.store(C_tile_ptrs, c_new, mask=(rm[:, None] < M) & (rn[None, :] < N))

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Performs two sequential operations:
      1) C = alpha * torch.mm(A, B) + beta * C
      2) C = alpha * torch.mm(C, C.T) + beta * C

    Args:
        A (torch.Tensor): (n, m)
        B (torch.Tensor): (m, p)
        C (torch.Tensor): (n, p)
        alpha (float)
        beta (float)

    Returns:
        torch.Tensor: Updated C after the two operations.
    """
    # First GEMM: C = alpha * (A*B) + beta * C
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, "A's second dimension must match B's first dimension."
    assert C.shape == (n, p), "C must be (n, p)."

    # Launch configuration
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        triton.cdiv(n, BLOCK_M),
        triton.cdiv(p, BLOCK_N),
    )

    # Ensure we are using contiguous tensors for simplicity
    A_c = A.contiguous()
    B_c = B.contiguous()
    C_c = C.contiguous()

    # Get strides for A, B, C
    stride_am, stride_ak = A_c.stride()
    stride_bk, stride_bn = B_c.stride()
    stride_cm, stride_cn = C_c.stride()

    # Convert strides to int64 for the kernel
    stride_am, stride_ak = stride_am, stride_ak
    stride_bk, stride_bn = stride_bk, stride_bn
    stride_cm, stride_cn = stride_cm, stride_cn

    _matmul_update_kernel[grid](
        A_c, B_c, C_c,
        n, p, m,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        alpha, beta,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    # Second GEMM: C = alpha * (C*C^T) + beta * C
    # For this, the shape of C*C^T is (n, n). We assume p == n for this to make sense.
    assert n == p, "For the symmetric update, C must be square (n == p)."
    n2, p2 = C_c.shape
    # Launch configuration for the second pass
    grid
