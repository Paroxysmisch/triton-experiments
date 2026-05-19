import torch
import triton
import triton.language as tl
import math

@triton.jit
def svd_reconstruct_kernel(
    u_ptr, s_ptr, vh_ptr, c_ptr,
    m, n, k,
    stride_um, stride_uk,
    stride_vhk, stride_vhn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    u_ptrs = u_ptr + offs_m[:, None] * stride_um + offs_k[None, :] * stride_uk
    s_ptrs = s_ptr + offs_k
    vh_ptrs = vh_ptr + offs_k[:, None] * stride_vhk + offs_n[None, :] * stride_vhn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for current_k in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        a = tl.load(u_ptrs, mask=(offs_m[:, None] < m) & (offs_k[None, :] < k - current_k * BLOCK_SIZE_K), other=0.0).to(tl.float32)
        s = tl.load(s_ptrs, mask=offs_k < k - current_k * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        b = tl.load(vh_ptrs, mask=(offs_k[:, None] < k - current_k * BLOCK_SIZE_K) & (offs_n[None, :] < n), other=0.0).to(tl.float32)
        
        a_scaled = a * s[None, :]
        accumulator += tl.dot(a_scaled, b)
        
        u_ptrs += BLOCK_SIZE_K * stride_uk
        s_ptrs += BLOCK_SIZE_K
        vh_ptrs += BLOCK_SIZE_K * stride_vhk
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

def fused_svd_reconstruct(A: torch.Tensor) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    U = U.contiguous()
    S = S.contiguous()
    Vh = Vh.contiguous()
    
    m, k = U.shape
    assert Vh.shape[0] == k, "Incompatible SVD matrices"
    n = Vh.shape[1]
    
    C = torch.empty((m, n), device=A.device, dtype=A.dtype)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    grid_m = triton.cdiv(m, BLOCK_SIZE_M)
    grid_n = triton.cdiv(n, BLOCK_SIZE_N)
    
    svd_reconstruct_kernel[(grid_m, grid_n, 1)](
        U, S, Vh, C,
        m, n, k,
        U.stride(0), U.stride(1),
        Vh.stride(0), Vh.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C
