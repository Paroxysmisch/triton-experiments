import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    x_ptr, y_ptr, z_ptr, m_size, n_size, k_size,
    m_block_size: tl.constexpr, n_block_size: tl.constexpr, k_block_size: tl.constexpr
):
    # pid = tl.program_id(axis)
    pid = tl.program_id(0)
    # num_pid_n = tl.cdiv(n_size, n_block_size)
    # num_pid_m = tl.cdiv(m_size, m_block_size)
    # num_pid_in_group = num_pid_n * num_pid_m
    # group_id = pid // num_pid_in_group
    # first_pid_m = group_id * num_pid_m
    # group_size_m = min(num_pid_m, (pid + 1) * num_pid_in_group // num_pid_n - first_pid_m)
    # pid_m = first_pid_m + (pid % group_size_m)
    # pid_n = (pid % num_pid_in_group) // group_size_m
    pid_m = pid // (n_size // n_block_size)
    pid_n = pid % (n_size // n_block_size)

    m_offset = pid_m * m_block_size
    n_offset = pid_n * n_block_size

    k_block_ptr = tl.arange(0, k_block_size)
    k_mask = k_block_ptr < k_size

    m_block_ptr = m_offset + tl.arange(0, m_block_size)
    m_mask = m_block_ptr < m_size

    n_block_ptr = n_offset + tl.arange(0, n_block_size)
    n_mask = n_block_ptr < n_size

    z = tl.zeros((m_block_size, n_block_size), dtype=tl.float32)
    for k in range(0, k_size, k_block_size):
        k_block_ptr = k + tl.arange(0, k_block_size)
        k_mask = k_block_ptr < k_size

        x_block_ptr = m_block_ptr[:, None] * n_size * k_size + k_block_ptr[None, :] * n_size + n_offset
        x = tl.load(x_ptr + x_block_ptr, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        y_block_ptr = k_block_ptr[:, None] * n_size + n_block_ptr[None, :]
        y = tl.load(y_ptr + y_block_ptr, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        z += tl.dot(x, y, z)

    z_mask = m_mask[:, None] & n_mask[None, :]
    z_block_ptr = m_offset + tl.arange(0, m_block_size)[:, None] * n_size + n_offset + tl.arange(0, n_block_size)[None, :]
    tl.store(z_ptr + z_block_ptr, z, mask=z_mask)

def matmul(x: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
    (m_size, k_size), (k_size, n_size) = x.shape, y.shape
    z = torch.empty((m_size, n_size), device=x.device, dtype=x.dtype)

    m_block_size, n_block_size, k_block_size = 128, 128, 32
    grid = (triton.cdiv(m_size, m_block_size) * triton.cdiv(n_size, n_block_size),)
    matmul_kernel[grid](
        x, y, z, m_size, n_size, k_size,
        m_block_size=m_block_size, n_block_size=n_block_size, k_block_size=k_block_size
    )

    return z
