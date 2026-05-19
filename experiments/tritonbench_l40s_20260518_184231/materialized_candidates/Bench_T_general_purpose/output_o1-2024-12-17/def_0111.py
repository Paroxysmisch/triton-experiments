import torch
import triton
import triton.language as tl

# -----------------------
# Triton Matrix Multiply Kernel
# -----------------------
# This kernel computes C = A x B, for matrices A (M x K) and B (K x N).
# Batch dimension is handled outside the kernel (e.g., via a Python loop).
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute block level offsets
    row_off = pid_m * BLOCK_M
    col_off = pid_n * BLOCK_N

    # Create pointers for block's start
    A_block_ptr = A_ptr + row_off * stride_am
    B_block_ptr = B_ptr + col_off * stride_bn

    # Create accumulators for partial results
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K chunks
    # We assume K is divisible by BLOCK_K for simplicity
    for k_off in range(0, K, BLOCK_K):
        # Load A and B blocks
        a = tl.load(
            A_block_ptr + (k_off * stride_ak),
            mask=(row_off + tl.arange(0, BLOCK_M) < M),
            other=0.0
        ).reshape((BLOCK_M, BLOCK_K))

        b = tl.load(
            B_block_ptr + (k_off * stride_bk),
            mask=(col_off + tl.arange(0, BLOCK_N) < N),
            other=0.0
        ).reshape((BLOCK_K, BLOCK_N))

        # Compute matmul for the current block
        acc += tl.dot(a, b)

    # Write back to C
    c_ptr = C_ptr + (row_off * stride_cm + col_off * stride_cn)
    tl.store(
        c_ptr,
        acc,
        mask=(
            (row_off + tl.arange(0, BLOCK_M))[:, None] < M
        ) & (
            (col_off + tl.arange(0, BLOCK_N))[None, :] < N
        )
    )

def _triton_matmul(A_block: torch.Tensor, B_block: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication using the Triton GPU kernel for a single batch.
    A_block: shape (M, K)
    B_block: shape (K, N)
    Returns: C: shape (M, N)
    """
    assert A_block.is_cuda and B_block.is_cuda, "Tensors must be on GPU."
    M, K = A_block.shape
    Kb, N = B_block.shape
    assert K == Kb, "Incompatible matrix dimensions for multiplication."

    # Allocate output
    C = torch.empty((M, N), dtype=A_block.dtype, device=A_block.device)

    # Grid: how many blocks along M and N dimensions
    BLOCK_M = 64  # you can tune these heuristics
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (
        ( (M + BLOCK_M - 1) // BLOCK_M ),
        ( (N + BLOCK_N - 1) // BLOCK_N )
    )

    _matmul_kernel[grid](
        A_block, B_block, C,
        M, N, K,
        A_block.stride(0), A_block.stride(1),
        B_block.stride(0), B_block.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    return C

def _apply_singular_values(U_k: torch.Tensor, S_k: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiply U_k (shape [m, k]) by singular values S_k (shape [k]) along dim=1.
    This handles real or complex types by broadcasting correctly.
    """
    # S_k: shape [k]
    # U_k: shape [m, k]
    # We can broadcast multiplication along the last dimension.
    return U_k * S_k.unsqueeze(0)

def low_rank_svd_approximation(A: torch.Tensor, k: int, *, full_matrices: bool = True, out=None) -> torch.Tensor:
    """
    Computes a rank-k approximation of a matrix A using its Singular Value Decomposition (SVD).
    Retains the top-k singular values and vectors.
    
    Args:
        A (Tensor): shape `(*, m, n)` where `*` is zero or more batch dimensions.
        k (int): Target rank, 1 <= k <= min(m, n).
        full_matrices (bool, optional): Controls whether to compute the full or reduced SVD. Default: True.
        out (Tensor, optional): Output tensor. Ignored if None. Default: None.

    Returns:
        Tensor: The rank-k approximation of A with the same shape as A.
    """
    # SVD decomposition
    # torch.linalg.svd supports real and complex dtypes, also supports batching
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)

    # Slice top-k components
    U_k = U[..., :k]          # shape (*, m, k)
    S_k = S[..., :k]          # shape (*, k)
    Vh_k = Vh[..., :k, :]     # shape (*, k, n)

    # Prepare output if needed
    if out is None:
        out_shape = A.shape
        out = A.new_empty(out_shape)

    # For each batch index, multiply U_k * diag(S_k) * Vh_k
    # We'll do: (U_k * S_k) @ Vh_k
    # We'll loop over any batch dimensions and call our Triton-based matmul.
    # This approach supports both real and complex dtypes.
    # We can flatten the batch, compute slice by slice, and reshape back.

    # Reshape into [batch_size, m, n], [batch_size, m, k], etc.
    batch_dims = A.shape[:-2]
    m, n = A.shape[-2], A.shape[-1]
    batch_size = 1
    for bd in batch_dims:
        batch_size *= bd

    # Flatten batch in a view
    U_k_view = U_k.reshape(batch_size, m, k)
    S_k_view = S_k.reshape(batch_size, k)
    Vh_k_view = Vh_k.reshape(batch_size, k, n)
    out_view = out.reshape(batch_size, m, n)

    for b_idx in range(batch_size):
        # Multiply U_k by singular values S_k
        U_k_s = _apply_singular_values(U_k_view[b_idx], S_k_view[b_idx])
        # Now matmul with Vh_k
        out_view[b_idx].copy_(_triton_matmul(U_k_s, Vh_k_view[b_idx]))

    return out
