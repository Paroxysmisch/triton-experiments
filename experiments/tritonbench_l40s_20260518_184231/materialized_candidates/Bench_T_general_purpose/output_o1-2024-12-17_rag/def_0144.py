import torch
import triton
import triton.language as tl

@triton.jit
def _scaled_matmul_add_kernel(
    A_ptr, B_ptr, C_ptr,
    alpha, beta,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs for dividing work across 2D tiles
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute the tile's corner indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create a K-range
    rk = tl.arange(0, BLOCK_K)
    
    # Pointers to A and B sub-blocks
    # Each block loads one sub-matrix from A, B
    a_ptrs = A_ptr + (rm[:, None] * stride_am) + (rk[None, :] * stride_ak)
    b_ptrs = B_ptr + (rk[:, None] * stride_bk) + (rn[None, :] * stride_bn)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension by BLOCK_K
    # Repeats until all partial sums are accumulated
    for k_block_start in range(0, K, BLOCK_K):
        # Load data from A, B
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (k_block_start + rk[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k_block_start + rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        # Multiply and accumulate
        acc += tl.dot(a, b)
        # Move ptrs to next K-block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Apply alpha scaling to computed block
    acc = alpha * acc

    # Now add beta*C
    c_ptrs = C_ptr + (rm[:, None] * stride_cm) + (rn[None, :] * stride_cn)
    c_old = tl.load(c_ptrs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    acc += beta * c_old

    # Write result back to C
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def matrix_multiply_and_row_dot(
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float,
    beta: float,
    C: torch.Tensor
) -> torch.Tensor:

    # Shapes
    # A: (n, m)
    # B: (m, p)
    # C: (n, p)
    n, m = A.shape
    assert B.shape[0] == m, "Incompatible shapes for A and B"
    p = B.shape[1]
    assert C.shape[0] == n and C.shape[1] == p, "Incompatible shape for C"

    # Make sure inputs are contiguous
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    # Cast to float32 if needed (Triton kernels typically expect float32)
    A32 = A.to(torch.float32)
    B32 = B.to(torch.float32)
    C32 = C.to(torch.float32)

    # Strides for row-major
    stride_am = A32.stride(0)
    stride_ak = A32.stride(1)
    stride_bk = B32.stride(0)
    stride_bn = B32.stride(1)
    stride_cm = C32.stride(0)
    stride_cn = C32.stride(1)

    # Grid: how many blocks along M, how many blocks along N
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        ( (n + BLOCK_M - 1) // BLOCK_M ),
        ( (p + BLOCK_N - 1) // BLOCK_N )
    )

    _scaled_matmul_add_kernel[grid](
        A32, B32, C32,
        alpha, beta,
        n, p, m,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    # Compute dot product of first two rows of updated C
    result = torch.dot(C32[0], C32[1])

    # Write updated result back to original C if needed (in float32)
    C.copy_(C32)

    # Return dot product as tensor
    return result
