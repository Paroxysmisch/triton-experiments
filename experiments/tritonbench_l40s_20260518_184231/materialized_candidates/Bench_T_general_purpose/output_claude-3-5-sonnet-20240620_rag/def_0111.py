import triton
import triton.language as tl
import torch

@triton.jit
def low_rank_svd_kernel(
    U_ptr, S_ptr, Vh_ptr, out_ptr,
    m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
    stride_um, stride_un,
    stride_s,
    stride_vm, stride_vn,
    stride_out_m, stride_out_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    Computes the rank-k approximation using pre-computed SVD components.
    A_k = U_k * Sigma_k * V_k^H
    """
    pid = tl.program_id(0)
    
    # Block indices
    bid_m = pid // (n // BLOCK_SIZE_N)
    bid_n = pid % (n // BLOCK_SIZE_N)
    
    # Initialize offsets
    offs_m = bid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = bid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute matrix multiplication U_k * (Sigma_k * V_k^H)
    for ki in range(0, k):
        # Load U_k block
        u_ptrs = U_ptr + offs_m * stride_um + ki * stride_un
        u = tl.load(u_ptrs)
        
        # Load singular value
        s = tl.load(S_ptr + ki * stride_s)
        
        # Load V_k^H block
        vh_ptrs = Vh_ptr + ki * stride_vm + offs_n * stride_vn
        vh = tl.load(vh_ptrs)
        
        # Accumulate U_k * (Sigma_k * V_k^H)
        acc += s * tl.outer(u, vh)
    
    # Store result
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, acc)

def low_rank_svd_approximation(A, k, *, full_matrices=True, out=None):
    """
    Computes rank-k approximation of matrix A using SVD.
    
    Args:
        A (Tensor): Input tensor of shape (*, m, n)
        k (int): Rank of approximation (1 <= k <= min(m,n))
        full_matrices (bool, optional): Whether to compute full or reduced SVD
        out (Tensor, optional): Output tensor
        
    Returns:
        Tensor: Rank-k approximation of A
    """
    # Input validation
    assert k >= 1, f"k must be at least 1, got {k}"
    *batch_dims, m, n = A.shape
    min_dim = min(m, n)
    assert k <= min_dim, f"k must be <= min(m,n), got k={k}, m={m}, n={n}"
    
    # Compute full SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    
    # Take top k components
    U_k = U[..., :, :k]
    S_k = S[..., :k]
    Vh_k = Vh[..., :k, :]
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(A)
    
    # Handle batched inputs
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Grid and block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    grid = (m * n + BLOCK_SIZE_M * BLOCK_SIZE_N - 1) // (BLOCK_SIZE_M * BLOCK_SIZE_N)
    
    # Launch kernel for each batch
    for batch_idx in range(batch_size):
        low_rank_svd_kernel[(grid,)](
            U_k[batch_idx].data_ptr(),
            S_k[batch_idx].data_ptr(),
            Vh_k[batch_idx].data_ptr(),
            out[batch_idx].data_ptr(),
            m, n, k,
            U_k.stride(-2), U_k.stride(-1),
            S_k.stride(-1),
            Vh_k.stride(-2), Vh_k.stride(-1),
            out.stride(-2), out.stride(-1),
            BLOCK_SIZE_M, BLOCK_SIZE_N
        )
    
    return out
