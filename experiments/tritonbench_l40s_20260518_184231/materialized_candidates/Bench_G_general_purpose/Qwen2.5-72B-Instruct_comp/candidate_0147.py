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
    OUTPUT_F16: tl.constexpr
):
    # Compute the program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_in_warp = tl.num_programs(axis=1)
    pid_m = pid // (num_pid_n * num_pid_in_warp)
    pid_n = (pid % (num_pid_n * num_pid_in_warp)) // num_pid_in_warp
    pid_k = (pid % (num_pid_n * num_pid_in_warp)) % num_pid_in_warp

    # Compute the block offsets
    rm = pid_m * BLOCK_M
    rn = pid_n * BLOCK_N
    rk = pid_k * BLOCK_K

    # Create block pointers for A, B, and C
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(rm, rk), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
    )
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(rk, rn), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1)
    )
    C_block_ptr = tl.make_block_ptr(
        base=C_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(rm, rn), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over the K dimension
    for k in range(0, K, BLOCK_K):
        # Load the blocks from A and B
        a = tl.load(A_block_ptr)
        b = tl.load(B_block_ptr)

        # Perform the matrix multiplication
        acc += tl.dot(a, b)

        # Advance the block pointers
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))

    # Convert the result to float16 if required
    if OUTPUT_F16:
        acc = acc.to(tl.float16)

    # Store the result in C
    tl.store(C_block_ptr, acc)

import triton
import triton.language as tl
import numpy as np
import torch

def wrapper_tma_load_store(M, N, K, num_warps, num_ctas, trans_a, trans_b, output_f16):
    # Generate random matrices A and B
    A = np.random.rand(M, K).astype(np.float16)
    B = np.random.rand(K, N).astype(np.float16)

    # Optionally transpose A and B
    if trans_a:
        A = A.T
    if trans_b:
        B = B.T

    # Allocate matrix C
    C = np.zeros((M, N), dtype=np.float16 if output_f16 else np.float32)

    # Convert matrices to Torch tensors
    A_torch = torch.tensor(A, device='cuda')
    B_torch = torch.tensor(B, device='cuda')
    C_torch = torch.tensor(C, device='cuda')

    # Define the block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    # Define the strides
    stride_am = A_torch.stride(0)
    stride_ak = A_torch.stride(1)
    stride_bk = B_torch.stride(0)
    stride_bn = B_torch.stride(1)
    stride_cm = C_torch.stride(0)
    stride_cn = C_torch.stride(1)

    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) * num_warps, 1, 1)
    matmul_tma_load_store[grid](
        A_torch, B_torch, C_torch,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K,
        output_f16
    )

    # Convert the result back to a NumPy array
    C = C_torch.cpu().numpy()

    return C
