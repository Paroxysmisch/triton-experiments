import torch
import triton
import triton.language as tl

@triton.jit
def iv_dependent_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * BLOCK_M
    group_size_m = min(num_pid_m - first_pid_m, BLOCK_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if TYPE == 'pre_load':
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

    for k in range(0, K, BLOCK_K):
        if TYPE == 'post_load':
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        accumulator += tl.dot(a, b)

        if TYPE != 'pre_load':
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        if TYPE == 'pre_load' and k + BLOCK_K < K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@torch.no_grad()
def iv_dependent_matmul_wrapper(a, b, c, block_m, block_n, block_k, type='post_load'):
    M, K = a.shape
    _, N = b.shape

    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions"
    assert c.shape == (M, N), "Output matrix has incorrect dimensions"

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    iv_dependent_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        TYPE=type,
    )

# Example usage
if __name__ == "__main__":
    M, K, N = 1024, 1024, 1024
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)

    iv_dependent_matmul_wrapper(a, b, c, block_m=64, block_n=64, block_k=32, type='post_load')

    # Verify result
    c_ref = torch.matmul(a, b)
    print(f"Max error: {torch.max(torch.abs(c - c_ref))}")
