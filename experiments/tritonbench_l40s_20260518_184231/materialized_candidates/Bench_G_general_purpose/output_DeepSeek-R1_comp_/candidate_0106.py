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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    type: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if type == "double_buffer":
        # Preload first block
        a = tl.load(a_ptrs, 
                   mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), 
                   other=0.0)
        b = tl.load(b_ptrs, 
                   mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), 
                   other=0.0)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

        # Main loop with double buffering
        for k in range(BLOCK_SIZE_K, K, BLOCK_SIZE_K):
            a_next = tl.load(a_ptrs, 
                            mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k)),
                            other=0.0)
            b_next = tl.load(b_ptrs,
                            mask=(offs_k[:, None] < (K - k)) & (offs_n[None, :] < N),
                            other=0.0)
            accumulator += tl.dot(a, b)
            a, b = a_next, b_next
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # Process last block
        accumulator += tl.dot(a, b)
    else:
        # Standard loading strategy
        for k in range(0, K, BLOCK_SIZE_K):
            a = tl.load(a_ptrs,
                       mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k)),
                       other=0.0)
            b = tl.load(b_ptrs,
                       mask=(offs_k[:, None] < (K - k)) & (offs_n[None, :] < N),
                       other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    # Write back result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)

def iv_dependent_matmul_wrapper(a, b, type='standard'):
    # Check device and dimensions
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA device"
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, "Incompatible dimensions"
    
    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Configure kernel parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    num_warps = 4
    num_stages = 3 if type == 'double_buffer' else 1
    
    # Compute grid size
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m * grid_n,)
    
    # Launch kernel
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
        num_warps=num_warps,
        num_stages=num_stages
    )
    return c

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, K = 1024, 1024, 1024
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    
    # Compute with standard loading
    triton_output = iv_dependent_matmul_wrapper(a, b, type='standard')
    
    # Compute with double buffering
    triton_output_db = iv_dependent_matmul_wrapper(a, b, type='double_buffer')
    
    # Verify correctness
    torch_output = torch.matmul(a, b)
    assert torch.allclose(triton_output, torch_output, atol=1e-2)
    assert torch.allclose(triton_output_db, torch_output, atol=1e-2)
    print("All close!")
