import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'split_k': 1},
            num_stages=3,
            num_warps=4
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'split_k': 2},
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'split_k': 4},
            num_stages=3,
            num_warps=8
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    split_k: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // split_k
    split_id = pid % split_k

    pid_m = group_id // num_pid_n
    pid_n = group_id % num_pid_n

    # Create block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize pointers with split_k consideration
    k_per_program = tl.cdiv(K, split_k)
    k_start = split_id * k_per_program
    k_end = tl.minimum(k_start + k_per_program, K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak)
    b_ptrs = b_ptr + ((k_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for current_k in range(k_start, k_end, BLOCK_SIZE_K):
        a_mask = (offs_m < M)[:, None] & ((current_k + offs_k) < K)[None, :]
        b_mask = ((current_k + offs_k) < K)[:, None] & (offs_n < N)[None, :]
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc += tl.dot(a, b, allow_tf32=False)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Write back result with appropriate masking
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    
    if split_k == 1:
        tl.store(c_ptrs, acc, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=c_mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D matrices"
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Dimension mismatch: {K} vs {K_}"
    
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * 
        triton.cdiv(N, meta['BLOCK_SIZE_N']) * 
        meta['split_k'],
        1,
        1
    )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

a = torch.randn(1024, 512, device='cuda', dtype=torch.float16)
b = torch.randn(512, 2048, device='cuda', dtype=torch.float16)

# Use Triton implementation
c_triton = triton_matmul(a, b)

# Verify against PyTorch implementation
c_ref = torch.matmul(a, b)
assert torch.allclose(c_triton, c_ref, atol=1e-2)
