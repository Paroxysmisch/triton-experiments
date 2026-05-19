import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------------
# Triton Kernel: _matmul_add_scale_kernel
# This kernel computes:
#   C[...] = alpha * (A x B) + beta * C[...]
# for blocks of the output.
# ---------------------------------------------------------------------------------
@triton.jit
def _matmul_add_scale_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    alpha, beta,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program ID (block index) for the M dimension and N dimension
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block indices for M and N dimensions
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # k-loop over shared dimension
    rk = tl.arange(0, BLOCK_K)
    # We'll move in steps of BLOCK_K along K dimension
    offs_k = tl.arange(0, BLOCK_K)
    offs_am = rm[:, None]    # shape [BLOCK_M, 1]
    offs_bn = rn[None, :]    # shape [1, BLOCK_N]

    # Accumulate partial sums
    for k_offs in range(0, K, BLOCK_K):
        # A_ptr offset
        a_ptrs = A_ptr + (offs_am * stride_am + (offs_k + k_offs) * stride_an)
        # B_ptr offset
        b_ptrs = B_ptr + ((offs_k + k_offs) * stride_bm + offs_bn * stride_bn)

        a_vals = tl.load(a_ptrs, mask=(rm < M)[:, None] & (rk + k_offs < K))
        b_vals = tl.load(b_ptrs, mask=(rn < N)[None, :] & (rk + k_offs < K))
        # Compute outer product for the chunk
        acc += tl.dot(a_vals.to(tl.float32), b_vals.to(tl.float32))

    # Load old values of C if in range
    c_ptrs = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    c_vals = tl.load(c_ptrs, mask=(rm < M)[:, None] & (rn < N)[None, :], other=0.0)

    # Perform the scaled addition: alpha * acc + beta * c_vals
    out = alpha * acc + beta * c_vals.to(tl.float32)

    # Store results back to C
    tl.store(c_ptrs, out, mask=(rm < M)[:, None] & (rn < N)[None, :])

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    """
    Computes a scaled matrix-matrix product, then calculates the dot product of the first two rows:
    1) C = alpha * (A @ B) + beta * C
    2) result = dot(C[0], C[1])

    Args:
        A (torch.Tensor): First input matrix of shape (n, m).
        B (torch.Tensor): Second input matrix of shape (m, p).
        alpha (float): Scalar multiplier for the matrix-matrix product.
        beta (float): Scalar multiplier for the input matrix C.
        C (torch.Tensor): Output matrix of shape (n, p) where the results are added.

    Returns:
        torch.Tensor: A scalar containing the dot product of the first two rows of C.
    """
    # Shapes
    n, m = A.shape
    m2, p = B.shape
    # Basic shape check (optional)
    assert m == m2, "Incompatible shapes for A and B."
    assert C.shape[0] >= 2 and C.shape[0] == n and C.shape[1] == p, "Incompatible shape for C or insufficient rows."

    # Grid dimensions: how many blocks needed in each dimension
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Number of program ids along M and N
    grid = (
        (n + BLOCK_M - 1) // BLOCK_M,
        (p + BLOCK_N - 1) // BLOCK_N
    )

    # Launch the Triton kernel
    _matmul_add_scale_kernel[grid](
        A, B, C,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        alpha, beta,
        n, p, m,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    # Compute dot product of first two rows of C
    result = torch.dot(C[0], C[1])
    return result
