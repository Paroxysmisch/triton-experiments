import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel_persistent(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_blocks = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M * (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_stages = (num_blocks + NUM_SMS - 1) // NUM_SMS

    for stage in range(num_stages):
        pid_stage = pid + stage * NUM_SMS
        if pid_stage >= num_blocks:
            break

        m = pid_stage // ((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) * BLOCK_SIZE_M
        n = pid_stage % ((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) * BLOCK_SIZE_N

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_SIZE_K):
            a = tl.load(a_ptr + m * stride_am + k * stride_ak + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :], mask=(m + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K), other=0.0)
            b = tl.load(b_ptr + k * stride_bk + n * stride_bn + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk + tl.arange(0, BLOCK_SIZE_N)[None, :], mask=(k + tl.arange(0, BLOCK_SIZE_K)[:, None] < K) & (n + tl.arange(0, BLOCK_SIZE_N)[None, :] < N), other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(tl.float16)
        tl.store(c_ptr + m * stride_cm + n * stride_cn + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_cm + tl.arange(0, BLOCK_SIZE_N)[None, :], c, mask=(m + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & (n + tl.arange(0, BLOCK_SIZE_N)[None, :] < N))

def matmul_persistent(a: torch.Tensor, b: torch.Tensor, M: int, N: int, K: int):
    assert a.shape == (M, K) and b.shape == (K, N), "Input matrices have incorrect shapes"
    assert a.dtype == b.dtype == torch.float16, "Input matrices must be of dtype torch.float16"

    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    NUM_SMS = 80  # Number of streaming multiprocessors

    matmul_kernel_persistent[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bn=b.stride(1), stride_bk=b.stride(0),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_SMS=NUM_SMS
    )

    return c

# Example usage
M, N, K = 1024, 1024, 1024
a = torch.randn((M, K), dtype=torch.float16, device='cuda')
b = torch.randn((K, N), dtype=torch.float16, device='cuda')
c = matmul_persistent(a, b, M, N, K)
print(c)
