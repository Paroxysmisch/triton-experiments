import torch
import triton
import triton.language as tl

@triton.jit
def svd_reconstruct_kernel(
    U_ptr, S_ptr, Vh_ptr, Out_ptr,
    m, n, k,
    stride_u_row, stride_u_col,
    stride_vh_row, stride_vh_col,
    stride_out_row, stride_out_col,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_idx in range(0, k, BLOCK_SIZE_K):
        k_offs = k_idx + offs_k
        mask_k = k_offs < k

        u_ptrs = U_ptr + offs_m[:, None] * stride_u_row + k_offs[None, :] * stride_u_col
        u = tl.load(u_ptrs, mask=(offs_m[:, None] < m) & mask_k, other=0.0)

        s_ptrs = S_ptr + k_offs
        s = tl.load(s_ptrs, mask=mask_k, other=0.0)

        scaled_u = u * s[None, :]

        vh_ptrs = Vh_ptr + k_offs[:, None] * stride_vh_row + offs_n[None, :] * stride_vh_col
        vh = tl.load(vh_ptrs, mask=(k_offs[:, None] < k) & (offs_n[None, :] < n), other=0.0)

        acc += tl.dot(scaled_u, vh, allow_tf32=False)

    out_ptrs = Out_ptr + offs_m[:, None] * stride_out_row + offs_n[None, :] * stride_out_col
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

def fused_svd_reconstruct(A: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 2, "Input must be a 2D tensor"
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    m, k = U.shape
    n = Vh.shape[1]
    
    A_reconstructed = torch.empty((m, n), dtype=A.dtype, device=A.device)
    
    if not U.is_contiguous():
        U = U.contiguous()
    if not S.is_contiguous():
        S = S.contiguous()
    if not Vh.is_contiguous():
        Vh = Vh.contiguous()
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_SIZE_M']) * triton.cdiv(n, meta['BLOCK_SIZE_N']),)
    
    svd_reconstruct_kernel[grid](
        U, S, Vh, A_reconstructed,
        m, n, k,
        U.stride(0), U.stride(1),
        Vh.stride(0), Vh.stride(1),
        A_reconstructed.stride(0), A_reconstructed.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return A_reconstructed
