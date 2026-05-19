import triton
import triton.language as tl
import kernel_utils

@triton.jit
def kernel(C, A, B, M, N, K,
           stride_cm, stride_cn,
           stride_am, stride_ak,
           stride_bk, stride_bn,
           BLOCK_M: tl.constexpr,
           BLOCK_N: tl.constexpr,
           BLOCK_K: tl.constexpr):
    # Triton kernel for matrix multiplication with extra elementwise operation
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = kernel_utils.mul(accumulator, accumulator)
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=offs_cm[:, None] < M & offs_cn[None, :] < N)

def int_matmul_kernel(a, b, c, config):
    # Calculate the grid size needed for the kernel execution
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = config.BLOCK_M, config.BLOCK_N, config.BLOCK_K
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch the kernel
    kernel[grid](c, a, b, M, N, K,
                 c.stride(0), c.stride(1),
                 a.stride(0), a.stride(1),
                 b.stride(0), b.stride(1),
                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                 num_warps=config.num_warps, num_stages=config.num_stages, num_ctas=config.num_ctas)

class Config:
    def __init__(self, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, num_ctas):
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_ctas = num_ctas

import torch

# Example matrices
M, K, N = 1024, 1024, 1024
a = torch.randn((M, K), device='cuda', dtype=torch.float32)
b = torch.randn((K, N), device='cuda', dtype=torch.float32)
c = torch.zeros((M, N), device='cuda', dtype=torch.float32)

# Configuration
config = Config(BLOCK_M=16, BLOCK_N=16, BLOCK_K=16, num_warps=4, num_stages=3, num_ctas=1)

# Launch the kernel
int_matmul_kernel(a, b, c, config)

# Verify the result
c_ref = (torch.mm(a, b) * torch.mm(a, b)).cpu().numpy()
c_test = c.cpu().numpy()
print("Maximum error:", np.max(np.abs(c_ref - c_test)))
