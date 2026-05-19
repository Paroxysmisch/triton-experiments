import torch
import triton
import triton.language as tl

@triton.jit
def svd_approximation_kernel(
    U_ptr, S_ptr, Vh_ptr, Ak_ptr,
    m, n, k,
    stride_U_row, stride_U_col,
    stride_S,
    stride_Vh_row, stride_Vh_col,
    stride_Ak_row, stride_Ak_col,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=U_ptr.dtype.element_ty)
    
    for i in range(0, k, BLOCK_SIZE_K):
        cols = i + offs_k
        mask_u = (offs_m[:, None] < m) & (cols[None, :] < k)
        u = tl.load(U_ptr + offs_m[:, None] * stride_U_row + cols[None, :] * stride_U_col, mask=mask_u, other=0.0)
        s = tl.load(S_ptr + cols, mask=cols < k, other=0.0)
        scaled_u = u * s[None, :]
        
        mask_vh = (cols[:, None] < k) & (offs_n[None, :] < n)
        vh = tl.load(Vh_ptr + cols[:, None] * stride_Vh_row + offs_n[None, :] * stride_Vh_col, mask=mask_vh, other=0.0)
        
        accum += tl.dot(scaled_u, vh, allow_tf32=True)
    
    mask_ak = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(Ak_ptr + offs_m[:, None] * stride_Ak_row + offs_n[None, :] * stride_Ak_col, accum, mask=mask_ak)

def low_rank_svd_approximation(A: torch.Tensor, k: int, *, full_matrices: bool = True, out: torch.Tensor = None) -> torch.Tensor:
    assert A.ndim >= 2, "A must have at least two dimensions"
    m, n = A.shape[-2], A.shape[-1]
    assert 1 <= k <= min(m, n), f"k must be between 1 and {min(m, n)}"
    
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    U_k = U[..., :k]
    S_k = S[..., :k]
    Vh_k = Vh[..., :k, :]
    
    if out is None:
        out = torch.empty_like(A)
    else:
        assert out.shape == A.shape, "out tensor must have the same shape as A"
    
    batch_dims = A.shape[:-2]
    num_batches = int(torch.prod(torch.tensor(batch_dims))) if batch_dims else 1
    U_flat = U_k.reshape(num_batches, m, k)
    S_flat = S_k.reshape(num_batches, k)
    Vh_flat = Vh_k.reshape(num_batches, k, n)
    out_flat = out.reshape(num_batches, m, n)
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(m, BLOCK_SIZE_M), triton.cdiv(n, BLOCK_SIZE_N))
    
    for i in range(num_batches):
        U_batch = U_flat[i]
        S_batch = S_flat[i]
        Vh_batch = Vh_flat[i]
        out_batch = out_flat[i]
        
        svd_approximation_kernel[grid](
            U_batch, S_batch, Vh_batch, out_batch,
            m, n, k,
            U_batch.stride(0), U_batch.stride(1),
            S_batch.stride(0),
            Vh_batch.stride(0), Vh_batch.stride(1),
            out_batch.stride(0), out_batch.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
    
    return out
