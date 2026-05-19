import triton
import triton.language as tl
import torch
import math

@triton.jit
def matmul_tma_load_store(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Matrix strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # Optional: output in float16
    use_fp16: tl.constexpr = False,
    # Group size
    GROUP_M: tl.constexpr = 8
):
    """
    Efficient matrix multiplication using TMA load/store operations
    """
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load blocks from A and B
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        
        # Compute block-level matrix multiplication
        acc += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store output
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn

    # Convert to float16 if requested
    if use_fp16:
        acc = acc.to(tl.float16)
    
    tl.store(c_ptrs, acc)

def wrapper_tma_load_store(M=1024, N=1024, K=1024, BLOCK_M=16, BLOCK_N=16, BLOCK_K=16, use_fp16=False):
    """
    Wrapper function to prepare inputs and call the kernel
    """
    # Create random input matrices
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16 if use_fp16 else torch.float32)

    # Compute grid size
    grid = lambda META: (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
    )

    # Run kernel
    matmul_tma_load_store[grid](
        a_ptr=a, 
        b_ptr=b,
        c_ptr=c,
        M=M, N=N, K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        use_fp16=use_fp16
    )

    return c
