import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, 4 * K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < (4 * K - k), other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < (K - k // 4), other=0)

        a0 = (a >> 6) & 0x03
        a1 = (a >> 4) & 0x03
        a2 = (a >> 2) & 0x03
        a3 = a & 0x03
        a0 = tl.where(a0 >= 2, a0 - 4, a0)
        a1 = tl.where(a1 >= 2, a1 - 4, a1)
        a2 = tl.where(a2 >= 2, a2 - 4, a2)
        a3 = tl.where(a3 >= 2, a3 - 4, a3)

        b0 = (b >> 6) & 0x03
        b1 = (b >> 4) & 0x03
        b2 = (b >> 2) & 0x03
        b3 = b & 0x03

        b_vec = tl.trans(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
        b0 = (b_vec >> 0) & 0xFF
        b1 = (b_vec >> 8) & 0xFF
        b2 = (b_vec >> 16) & 0xFF
        b3 = (b_vec >> 24) & 0xFF

        b0 = tl.reshape(b0, (BLOCK_SIZE_K, BLOCK_SIZE_N))
        b1 = tl.reshape(b1, (BLOCK_SIZE_K, BLOCK_SIZE_N))
        b2 = tl.reshape(b2, (BLOCK_SIZE_K, BLOCK_SIZE_N))
        b3 = tl.reshape(b3, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        acc += tl.dot(a0, b0, allow_tf32=False) + tl.dot(a1, b1, allow_tf32=False) \
             + tl.dot(a2, b2, allow_tf32=False) + tl.dot(a3, b3, allow_tf32=False)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 4) * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D"
    M, K4 = a.shape
    K, N = b.shape
    assert K4 == 4 * K, f"a.shape[1] must be 4*b.shape[0], got {K4} and {K}"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    c = torch.empty((M, N), dtype=torch.int32, device=a.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    matmul_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return c
