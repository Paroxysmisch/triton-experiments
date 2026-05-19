import triton
import triton.language as tl
import torch

@triton.jit
def matmul_tma_load_store(A_ptr, B_ptr, C_ptr,
                          M, N, K,
                          stride_am, stride_ak,
                          stride_bk, stride_bn,
                          stride_cm, stride_cn,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          OUTPUT_FLOAT16: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A = tl.load(A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak), mask=(offs_am[:, None] < M) & (offs_k[None, :] < K))
    B = tl.load(B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn), mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N))

    C = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        A_block = tl.load(A_ptr + (offs_am[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak), mask=(offs_am[:, None] < M) & ((k + offs_k)[None, :] < K))
        B_block = tl.load(B_ptr + ((k + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn), mask=((k + offs_k)[:, None] < K) & (offs_bn[None, :] < N))
        C += tl.dot(A_block, B_block)

    if OUTPUT_FLOAT16:
        C = C.to(tl.float16)

    tl.store(C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn), C, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

def wrapper_tma_load_store(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, OUTPUT_FLOAT16=False):
    # Prepare random input matrices
    A = torch.randn((M, K), dtype=torch.float16, device='cuda')
    B = torch.randn((K, N), dtype=torch.float16, device='cuda')
    C = torch.empty((M, N), dtype=torch.float16 if OUTPUT_FLOAT16 else torch.float32, device='cuda')

    # Define strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Launch the Triton kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_tma_load_store[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        OUTPUT_FLOAT16=OUTPUT_FLOAT16
    )

    return C

# Example usage
M, N, K = 128, 128, 128
BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 16
C = wrapper_tma_load_store(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, OUTPUT_FLOAT16=True)
print(C)
