import torch
import triton
import triton.language as tl
from triton.runtime import driver

@triton.jit
def matmul_kernel(
    x_ptr, y_ptr, z_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_yk, stride_yn,
    stride_zm, stride_zn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    y_ptrs = y_ptr + (offs_k[:, None] * stride_yk + offs_yn[None, :] * stride_yn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x_block = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        y_block = tl.load(y_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(x_block, y_block.to(tl.float16))
        x_ptrs += BLOCK_SIZE_K * stride_xk
        y_ptrs += BLOCK_SIZE_K * stride_yk

    offs_zm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_zn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    z_ptrs = z_ptr + (offs_zm[:, None] * stride_zm + offs_zn[None, :] * stride_zn)
    mask = (offs_zm[:, None] < M) & (offs_zn[None, :] < N)
    tl.store(z_ptrs, accumulator, mask=mask)

def matmul(x, y, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=8):
    assert x.shape[1] == y.shape[0], "Matrix dimensions incompatible for multiplication"
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    M, K = x.shape
    K_, N = y.shape
    assert K == K_, "Inner dimensions must match"
    z = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        x, y, z,
        M, N, K,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        z.stride(0), z.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    return z
