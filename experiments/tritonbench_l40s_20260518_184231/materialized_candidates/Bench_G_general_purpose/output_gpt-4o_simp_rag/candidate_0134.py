import triton
import triton.language as tl

class Config:
    def __init__(self, num_warps=4, num_stages=2, num_ctas=1):
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_ctas = num_ctas

@triton.jit
def matmul_kernel_with_block_pointers(C, A, B, M, N, K,
                                      stride_cm, stride_cn,
                                      stride_am, stride_ak,
                                      stride_bk, stride_bn,
                                      BLOCK_M: tl.constexpr,
                                      BLOCK_N: tl.constexpr,
                                      BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator)

@triton.jit
def scaled_matmul_kernel_with_block_pointers(C, A, B, scales1, M, N, K,
                                             stride_cm, stride_cn,
                                             stride_am, stride_ak,
                                             stride_bk, stride_bn,
                                             stride_scales1,
                                             BLOCK_M: tl.constexpr,
                                             BLOCK_N: tl.constexpr,
                                             BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    scale = tl.load(scales1 + pid_m * BLOCK_M * stride_scales1, mask=offs_am < M, other=1)
    accumulator *= scale[:, None]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator)

def int_matmul_kernel(C, A, B, M, N, K, config: Config):
    grid = (triton.cdiv(M, config.num_ctas), triton.cdiv(N, config.num_ctas))
    matmul_kernel_with_block_pointers[grid](
        C, A, B, M, N, K,
        C.stride(0), C.stride(1),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_M=config.num_ctas, BLOCK_N=config.num_ctas, BLOCK_K=config.num_warps
    )

def int_scaled_matmul_kernel(C, A, B, scales1, M, N, K, config: Config):
    grid = (triton.cdiv(M, config.num_ctas), triton.cdiv(N, config.num_ctas))
    scaled_matmul_kernel_with_block_pointers[grid](
        C, A, B, scales1, M, N, K,
        C.stride(0), C.stride(1),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        scales1.stride(0),
        BLOCK_M=config.num_ctas, BLOCK_N=config.num_ctas, BLOCK_K=config.num_warps
    )
