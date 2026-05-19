import triton
import triton.language as tl
import torch

@triton.jit
def _diag_mul_kernel(
    U_ptr, S_ptr,
    M: tl.constexpr, K: tl.constexpr,
    strideUM: tl.constexpr, strideUK: tl.constexpr, strideS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """
    In-place multiply a matrix U by the diagonal entries in S.
    U has shape [M, K], S has shape [K].
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    mask_m = rm < M
    mask_k = rk < K

    # Compute pointer offsets
    U_offset = rm[:, None] * strideUM + rk[None, :] * strideUK
    S_offset = rk

    # Load
    U_val = tl.where(mask_m[:, None] & mask_k[None, :],
                     tl.load(U_ptr + U_offset, mask=mask_m[:, None] & mask_k[None, :], other=0.0), 0.0)
    S_val = tl.where(mask_k, tl.load(S_ptr + S_offset, mask=mask_k, other=0.0), 0.0)

    # In-place multiply by diagonal
    U_val = U_val * S_val[None, :]

    # Store
    tl.store(U_ptr + U_offset, U_val, mask=mask_m[:, None] & mask_k[None, :])


@triton.jit
def _block_mm_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    strideAM: tl.constexpr, strideAK: tl.constexpr,
    strideBK: tl.constexpr, strideBN: tl.constexpr,
    strideCM: tl.constexpr, strideCN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """
    Computes C = A x B for blocks of size [BLOCK_SIZE_M, BLOCK_SIZE_N].
    A is [M, K], B is [K, N], C is [M, N].
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # Loop over K dimension
    for k_block_start in range(0, K, BLOCK_SIZE_K):
        rk = tl.arange(0, BLOCK_SIZE_K) + k_block_start
        # Load A block
        A_offset = rm[:, None] * strideAM + rk[None, :] * strideAK
        A_fragment = tl.where(
            (rm[:, None] < M) & (rk[None, :] < K),
            tl.load(A_ptr + A_offset, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0),
            0.0
        )
        # Load B block
        B_offset = rk[:, None] * strideBK + rn[None, :] * strideBN
        B_fragment = tl.where(
            (rk[:, None] < K) & (rn[None, :] < N),
            tl.load(B_ptr + B_offset, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0),
            0.0
        )
        # Matmul accumulate
        acc += tl.dot(A_fragment, B_fragment)

    # Store result to C
    C_offset = rm[:, None] * strideCM + rn[None, :] * strideCN
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr + C_offset, acc, mask=mask)


def low_rank_svd_approximation(A, k, *, full_matrices=True, out=None) -> torch.Tensor:
    """
    Computes the rank-k approximation of A using its SVD. 
    Retains the top-k singular values and corresponding singular vectors.

    Args:
        A (Tensor): Tensor of shape (*, m, n)
        k (int): Rank of approximation (1 <= k <= min(m, n))
        full_matrices (bool, optional): If True, computes full SVD. Default: True.
        out (Tensor, optional): Output tensor. Ignored if None.

    Returns:
        Tensor: Low-rank approximation of A with shape (*, m, n).
    """
    # Compute SVD via PyTorch (supports real/complex, double/float)
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    # Slice out top k
    U_k = U[..., :k]
    S_k = S[..., :k]
    Vh_k = Vh[..., :k, :]

    # We will do: U_k = U_k * diag(S_k)
    # Then A_k = U_k @ Vh_k

    # Flatten batch dimensions for Triton kernel calls if needed
    original_shape = U_k.shape
    # shape is (*batch, m, k), but we handle each batch separately in a loop
    batch_dims = original_shape[:-2]
    m, k_ = original_shape[-2], original_shape[-1]
    # S_k shape: (*batch, k)
    # Vh_k shape: (*batch, k, n)

    # Output shape is (*batch, m, n)
    n = Vh_k.shape[-1]

    # Prepare out
    if out is None:
        out = A.new_empty((*batch_dims, m, n))

    # Convert all involved to float32 for Triton kernels if feasible
    # (Triton can handle float16/float32 well, double/complex need fallback or specialized approach)
    # For simplicity, fallback to standard matmul if not float32 or float16.
    # Then do the block-by-block approach for each batch item.
    if A.dtype not in (torch.float16, torch.float32):
        # Fallback: do the computation in
