import torch
import triton
import triton.language as tl

@triton.jit
def iv_dependent_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    type: tl.constexpr,  # Loading strategy type
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs_base = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs_base = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Double loop strategy: outer and inner K loops for different loading
    for k_outer in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_inner = k_outer * BLOCK_SIZE_K
        a_ptrs = a_ptrs_base + k_inner * stride_ak
        b_ptrs = b_ptrs_base + k_inner * stride_bk

        # Load blocks based on strategy type
        if type == "preload":
            # Preload next block while computing current
            a_next_ptrs = a_ptrs + BLOCK_SIZE_K * stride_ak
            b_next_ptrs = b_ptrs + BLOCK_SIZE_K * stride_bk
            a = tl.load(a_ptrs, mask=(k_inner + offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(k_inner + offs_k[:, None] < K), other=0.0)
            a_next = tl.load(a_next_ptrs, mask=(k_inner + BLOCK_SIZE_K + offs_k[None, :] < K), other=0.0)
            b_next = tl.load(b_next_ptrs, mask=(k_inner + BLOCK_SIZE_K + offs_k[:, None] < K), other=0.0)
            accumulator += tl.dot(a, b)
            accumulator += tl.dot(a_next, b_next)
            k_inner += BLOCK_SIZE_K * 2  # Process two blocks per iteration
        else:  # Default strategy
            a = tl.load(a_ptrs, mask=(k_inner + offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(k_inner + offs_k[:, None] < K), other=0.0)
            accumulator += tl.dot(a, b)
            k_inner += BLOCK_SIZE_K

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)

def iv_dependent_matmul_wrapper(a, b, type: str = "default"):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )

    # Configure stages and warps based on type
    config = {'num_stages': 3, 'num_warps': 8}
    if type == "preload":
        config = {'num_stages': 4, 'num_warps': 4}

    iv_dependent_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        type=type,
        **config
    )
    return c

# Example usage
a = torch.randn(512, 256, device='cuda', dtype=torch.float16)
b = torch.randn(256, 384, device='cuda', dtype=torch.float16)
output = iv_dependent_matmul_wrapper(a, b, type="preload")
