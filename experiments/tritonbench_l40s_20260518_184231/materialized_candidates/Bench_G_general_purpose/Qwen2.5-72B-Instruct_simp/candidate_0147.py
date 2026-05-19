import triton
import triton.language as tl

@triton.jit
def matmul_tma_load_store(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    OUTPUT_FLOAT16: tl.constexpr
):
    # Compute the block ID in the grid
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size = num_pid_in_group
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = (pid % group_size) // num_pid_m

    # Compute the block offsets
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load the block pointers
    A = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(offs_am, offs_k),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    B = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(offs_k, offs_bn),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)
    )

    # Iterate over the K dimension
    for k in range(0, K, BLOCK_K):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A = tl.advance(A, (0, BLOCK_K))
        B = tl.advance(B, (BLOCK_K, 0))

    # Store the result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(offs_cm, offs_cn),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    if OUTPUT_FLOAT16:
        acc = acc.to(tl.float16)
    tl.store(C, acc)

import triton
import triton.language as tl
import torch

def wrapper_tma_load_store(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, OUTPUT_FLOAT16=False):
    # Generate random input matrices
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    C = torch.empty((M, N), device='cuda', dtype=torch.float16 if OUTPUT_FLOAT16 else torch.float32)

    # Compute the grid size
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # Launch the kernel
    matmul_tma_load_store[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        OUTPUT_FLOAT16
    )

    return C

# Example usage
M, N, K = 1024, 1024, 1024
BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 16
C = wrapper_tma_load_store(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, OUTPUT_FLOAT16=True)
print(C)
