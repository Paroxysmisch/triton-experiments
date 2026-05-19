import torch
import triton
import triton.language as tl
from triton.runtime import driver

@triton.jit
def matmul_kernel(
    x_ptr, y_ptr, z_ptr,
    m_size, n_size, k_size,
    stride_xm, stride_xk,
    stride_yk, stride_yn,
    stride_zm, stride_zn,
    m_block_size: tl.constexpr, n_block_size: tl.constexpr, k_block_size: tl.constexpr,
    group_size_m: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m_size, m_block_size)
    num_pid_n = tl.cdiv(n_size, n_block_size)
    num_pid_in_group = group_size_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size_m
    group_size_m_used = min(num_pid_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + (pid % group_size_m_used)
    pid_n = (pid % num_pid_in_group) // group_size_m_used

    offs_xm = (pid_m * m_block_size + tl.arange(0, m_block_size)) % m_size
    offs_yn = (pid_n * n_block_size + tl.arange(0, n_block_size)) % n_size
    offs_k = tl.arange(0, k_block_size)

    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    y_ptrs = y_ptr + (offs_k[:, None] * stride_yk + offs_yn[None, :] * stride_yn)

    z_acc = tl.zeros((m_block_size, n_block_size), dtype=tl.float32)

    for k in range(0, tl.cdiv(k_size, k_block_size)):
        x_block = tl.load(
            x_ptrs,
            mask=offs_k[None, :] < k_size - k * k_block_size,
            other=0.0
        )
        y_block = tl.load(
            y_ptrs,
            mask=offs_k[:, None] < k_size - k * k_block_size,
            other=0.0
        )
        z_acc += tl.dot(x_block, y_block.to(tl.float16))
        x_ptrs += k_block_size * stride_xk
        y_ptrs += k_block_size * stride_yk

    offs_zm = pid_m * m_block_size + tl.arange(0, m_block_size)
    offs_zn = pid_n * n_block_size + tl.arange(0, n_block_size)
    z_ptrs = z_ptr + stride_zm * offs_zm[:, None] + stride_zn * offs_zn[None, :]
    z_mask = (offs_zm[:, None] < m_size) & (offs_zn[None, :] < n_size)
    tl.store(z_ptrs, z_acc, mask=z_mask)

def matmul(x, y):
    assert x.shape[1] == y.shape[0], "Inner dimensions must match for matrix multiplication"
    assert x.is_contiguous(), "Matrix x must be contiguous"
    assert y.is_contiguous(), "Matrix y must be contiguous"

    m_size, k_size = x.shape
    k_size_y, n_size = y.shape

    z = torch.empty((m_size, n_size), device=x.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(m_size, META['m_block_size']) * triton.cdiv(n_size, META['n_block_size']),
    )

    matmul_kernel[grid](
        x, y, z,
        m_size, n_size, k_size,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        z.stride(0), z.stride(1)
    )
    return z
