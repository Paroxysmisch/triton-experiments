import torch
import triton
import triton.language as tl
import time

@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'type': 'ic'}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'type': 'kc'}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'type': 'ic'}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'type': 'kc'}, num_stages=4, num_warps=8),
], key=['M', 'N', 'K'])
@triton.jit
def iv_dependent_matmul_kernel(a_ptr, b_ptr, c_ptr,
                               M, N, K,
                               stride_am, stride_ak,
                               stride_bk, stride_bn,
                               stride_cm, stride_cn,
                               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, type: tl.constexpr):
    pid = tl.program_id(axis=0)
    i = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    j = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    i_m = i % M
    i_k = i // M
    j_k = j % K
    j_n = j // K
    k = i_k * K + j_k

    a_ptrs = a_ptr + i_m[:, None] * stride_am + k[None, :] * stride_ak
    b_ptrs = b_ptr + k[:, None] * stride_bk + j_n[None, :] * stride_bn
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if type == 'ic':
        for i_a in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            if (i_a * BLOCK_SIZE_K + j_k) < K:
                a = tl.load(a_ptrs)
            else:
                a = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

            if (i_k * BLOCK_SIZE_K + k - i_a * BLOCK_SIZE_K) < K:
                b = tl.load(b_ptrs)
            else:
                b = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

            c += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    elif type == 'kc':
        for j_b in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            if (i_k * BLOCK_SIZE_K + j_b * BLOCK_SIZE_K + k) < K:
                a = tl.load(a_ptrs)
            else:
                a = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

            if (j_b * BLOCK_SIZE_K + j_n) < K:
                b = tl.load(b_ptrs)
            else:
                b = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

            c += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + i_m[:, None] * stride_cm + j_n[None, :] * stride_cn
    tl.store(c_ptrs, c)

def iv_dependent_matmul_wrapper(device, K, M, N, case, type):
    torch.manual_seed(0)
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    if case == 'warmup':
        iv_dependent_matmul_kernel[grid](a, b, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), 1, 1, type=type)
    elif case == 'measurement':
        c = torch.empty((M, N), device=device, dtype=torch.float16)
        start = time.monotonic_ns()
        iv_dependent_matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), type=type)
        print("%s,%d,%d,%d,%d" % (type, K, M, N, time.monotonic_ns() - start))

triton_output = {}
device = torch.device('cuda')

K, M, N = 512, 2048, 2048
iv_dependent_matmul_wrapper(device, K, M, N, 'warmup', 'ic')
iv_dependent_matmul_wrapper(device, K, M, N, 'measurement', 'ic')
iv_dependent_matmul_wrapper(device, K, M, N, 'measurement', 'kc')
