import torch
import triton
import triton.language as tl

# ----------------------------------
# Triton Kernel for Matrix Multiplication
#  Computes: Out = X @ W
#  where X: (M, K), W: (K, N), and Out: (M, N).
# ----------------------------------
@triton.jit
def _matmul_kernel(
    X_ptr,      # *F32
    W_ptr,      # *F32
    Out_ptr,    # *F32
    M,          # int
    N,          # int
    K,          # int
    stride_xm,  # int
    stride_xk,  # int
    stride_wk,  # int
    stride_wn,  # int
    stride_om,  # int
    stride_on,  # int
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create a range of indices for the block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Create pointers for X and W block
    X_ptrs = X_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk)
    W_ptrs = W_ptr + (rk[:, None] * stride_wk + rn[None, :] * stride_wn)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # k = 0 to K in steps of BLOCK_K
    for k_block_start in range(0, K, BLOCK_K):
        # Load X and W tiles
        x = tl.load(X_ptrs, mask=(rm[:, None] < M) & (k_block_start + rk[None, :] < K), other=0.0)
        w = tl.load(W_ptrs, mask=(k_block_start + rk[:, None] < K) & (rn[None, :] < N), other=0.0)

        # Compute partial matmul
        acc += tl.dot(x, w)

        # Update pointers to the next block
        X_ptrs += BLOCK_K * stride_xk
        W_ptrs += BLOCK_K * stride_wk

    # Write back the result
    Out_ptrs = Out_ptr + (rm[:, None] * stride_om + rn[None, :] * stride_on)
    tl.store(Out_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

# ----------------------------------
# Helper function for Triton-based MatMul
# ----------------------------------
def triton_matmul(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Computes X @ W using a Triton-based matrix multiplication kernel.
    Shapes:
        X: (M, K)
        W: (K, N)
    Returns:
        Out: (M, N)
    """
    assert X.is_cuda and W.is_cuda, "Input tensors must be on CUDA device."
    M, K = X.shape
    K2, N = W.shape
    assert K == K2, "Incompatible dimensions for matmul."

    # Allocate output
    Out = torch.empty((M, N), device=X.device, dtype=X.dtype)

    # Grid dimensions
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        ( (M + BLOCK_M - 1) // BLOCK_M ),
        ( (N + BLOCK_N - 1) // BLOCK_N ),
    )

    # Launch Triton kernel
    _matmul_kernel[grid](
        X, W, Out,
        M, N, K,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return Out

# ----------------------------------
# Wrapper: fused_svd_reconstruct(A: Tensor) -> Tensor
# ----------------------------------
def fused_svd_reconstruct(A: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs the input matrix A using its SVD decomposition.
    A = U @ diag(S) @ V^H
    Returns the reconstructed matrix approximating the original A.
    
    Args:
        A (torch.Tensor): Input matrix of shape (m, n).
    Returns:
        torch.Tensor: Reconstructed matrix of shape (m, n).
    """

    # Compute SVD using PyTorch
    # shape(A) = (m, n)
    # shape(U) = (m, k), shape(S) = (k,), shape(Vh) = (k, n), where k = min(m, n)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # We need to form U * diag(S). This can be done by elementwise multiplying
    # each column of U by the singular values S.
    # shape(U) = (m, k), shape(S) = (k,)
    # This step is done via broadcasting in PyTorch:
    US = U * S.unsqueeze(0)

    # We'll compute the final product: (US) @ V^H with a Triton kernel
    A_reconstructed = triton_matmul(US, Vh)

    return A_reconstructed
