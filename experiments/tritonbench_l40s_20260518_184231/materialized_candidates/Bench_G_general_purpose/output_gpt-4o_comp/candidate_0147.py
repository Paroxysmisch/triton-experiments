import triton
import triton.language as tl

@triton.jit
def matmul_tma_load_store(A_ptr, B_ptr, C_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, OUTPUT_F16: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    m_idx = pid // grid_n
    n_idx = pid % grid_n

    # Block pointers
    a_ptr = tl.make_block_ptr(A_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(m_idx * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(0, 1))
    b_ptr = tl.make_block_ptr(B_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, n_idx * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    c_ptr = tl.make_block_ptr(C_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(m_idx * BLOCK_M, n_idx * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(0, 1))

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load and compute
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        acc += tl.dot(a, b)
        a_ptr = a_ptr + BLOCK_K * stride_ak
        b_ptr = b_ptr + BLOCK_K * stride_bk

    # Convert and store
    if OUTPUT_F16:
        acc = acc.to(tl.float16)
    tl.store(c_ptr, acc)

import torch
import triton

def wrapper_tma_load_store(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_ctas, transpose_a=False, transpose_b=False, output_f16=False):
    # Initialize random matrices A and B
    A = torch.randn((M, K), dtype=torch.float16, device='cuda')
    B = torch.randn((K, N), dtype=torch.float16, device='cuda')
    
    # Optionally transpose matrices
    if transpose_a:
        A = A.t()
    if transpose_b:
        B = B.t()

    # Allocate matrix C
    C = torch.empty((M, N), dtype=torch.float16 if output_f16 else torch.float32, device='cuda')

    # Strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_tma_load_store[grid](A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M, BLOCK_N, BLOCK_K, output_f16)

    return C

# Example usage
M, N, K = 128, 128, 128
BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
C = wrapper_tma_load_store(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4, num_ctas=1, transpose_a=False, transpose_b=False, output_f16=True)
print(C)
