import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    C_ptr, A_ptr, B_ptr,
    M, N, K,
    stride_cm, stride_cn,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    ALPHA: tl.constexpr = 0.01
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K
        offs_k = k_offs + tl.arange(0, BLOCK_SIZE_K)
        
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        accumulator += tl.dot(a, b)
    
    if ACTIVATION == "leaky_relu":
        accumulator = tl.where(accumulator > 0, accumulator, ALPHA * accumulator)
    
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=mask_c)

def matmul(a, b, activation=None, alpha=0.01):
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Matrix dimensions incompatible for multiplication: {K} vs {K_}"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    stride_cm, stride_cn = c.stride()
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    ACTIVATION = "none"
    if activation == "leaky_relu":
        ACTIVATION = "leaky_relu"
    elif activation is not None:
        raise ValueError(f"Activation '{activation}' is not supported")
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    matmul_kernel[grid](
        c, a, b,
        M, N, K,
        stride_cm, stride_cn,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        ACTIVATION, alpha
    )
    
    return c
