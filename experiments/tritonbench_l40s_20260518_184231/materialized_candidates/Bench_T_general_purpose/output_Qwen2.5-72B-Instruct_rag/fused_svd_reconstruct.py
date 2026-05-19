import torch
import triton
import triton.language as tl

# Kernel function: Matrix multiplication C = A @ B
@triton.jit
def matmul_kernel(
    A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_warp = BLOCK_SIZE_M * BLOCK_SIZE_N // 32
    num_warp_m = tl.cdiv(BLOCK_SIZE_M, 32)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    warp_id = pid % num_pid_in_warp
    wm = warp_id % num_warp_m
    wn = warp_id // num_warp_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_am
    offs_ak = tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak
    offs_bk = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_bk
    offs_bn = tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bn
    offs_cm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_cm
    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_cn

    A = A + (offs_am + offs_ak)
    B = B + (offs_bk + offs_bn)
    C = C + (offs_cm + offs_cn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A + k * stride_ak, mask=k + offs_ak < K, other=0.0)
        b = tl.load(B + k * stride_bk, mask=k + offs_bk < K, other=0.0)
        acc += tl.dot(a, b)

    tl.store(C, acc, mask=offs_cm < M and offs_cn < N)

def fused_svd_reconstruct(A: torch.Tensor) -> torch.Tensor:
    # Compute the SVD of A
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Convert S to a diagonal matrix
    S_diag = torch.diag(S)
    
    # Perform the matrix multiplication U @ S_diag @ Vh using Triton
    M, K = U.shape
    K, N = Vh.shape
    
    # Prepare the output tensor
    A_reconstructed = torch.empty_like(A)
    
    # Determine the block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Determine the grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # Launch the Triton kernel
    matmul_kernel[grid](
        U, S_diag, A_reconstructed, M, N, K,
        U.stride(0), U.stride(1), S_diag.stride(0), S_diag.stride(1), A_reconstructed.stride(0), A_reconstructed.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    # Perform the second matrix multiplication A_reconstructed @ Vh
    M, K = A_reconstructed.shape
    K, N = Vh.shape
    
    # Prepare the final output tensor
    A_final = torch.empty_like(A)
    
    # Determine the grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # Launch the Triton kernel
    matmul_kernel[grid](
        A_reconstructed, Vh, A_final, M, N, K,
        A_reconstructed.stride(0), A_reconstructed.stride(1), Vh.stride(0), Vh.stride(1), A_final.stride(0), A_final.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return A_final
