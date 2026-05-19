import triton
import triton.language as tl

@triton.jit
def matmul4_kernel(
    A, B, C, scales, zeros,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_block = tl.load(a_ptrs)
        b_block = tl.load(b_ptrs)

        b_block = b_block.to(tl.int4)
        b_block = b_block.to(tl.int32)
        b_block = (b_block - zeros) * scales

        accumulator += tl.dot(a_block, b_block)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    c_ptrs = C + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

import torch

def matmul_dequantize_int4_gptq(A, B, scales, zeros):
    M, K = A.shape
    _, N = B.shape

    # Ensure A and B are in the correct format
    A = A.to(torch.float16)
    B = B.to(torch.int32)
    scales = scales.to(torch.float16)
    zeros = zeros.to(torch.float16)

    # Allocate output tensor
    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    # Define grid and block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    # Launch the kernel
    matmul4_kernel[grid](
        A, B, C, scales, zeros,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M
    )

    return C

def quantize_int4(X, group_size=128):
    N, K = X.shape
    scales = torch.empty((N, K // group_size), dtype=torch.float16, device=X.device)
    zeros = torch.empty((N, K // group_size), dtype=torch.float16, device=X.device)
    B = torch.empty((N, K // 2), dtype=torch.int32, device=X.device)

    for i in range(0, K, group_size):
        x = X[:, i:i + group_size]
        max_val = torch.max(x.abs(), dim=1, keepdim=True).values
        scales[:, i // group_size] = max_val / 7
        zeros[:, i // group_size] = 0

        x = x / scales[:, i // group_size]
        x = x.round().to(torch.int8)

        for j in range(0, group_size, 2):
            b = (x[:, j].to(torch.int32) & 0x0F) | ((x[:, j + 1].to(torch.int32) & 0x0F) << 4)
            B[:, i // 2 + j // 2] = b

    return B, scales, zeros

import torch

# Example input matrices
M, K, N = 1024, 1024, 1024
A = torch.randn((M, K), dtype=torch.float16, device='cuda')
B, scales, zeros = quantize_int4(torch.randn((K, N), dtype=torch.float16, device='cuda'))

# Perform matrix multiplication
C = matmul_dequantize_int4_gptq(A, B, scales, zeros)

print(C)
