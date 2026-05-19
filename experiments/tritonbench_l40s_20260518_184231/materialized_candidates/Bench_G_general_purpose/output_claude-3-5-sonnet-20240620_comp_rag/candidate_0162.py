import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    m_size, n_size, k_size,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    m_block_size: tl.constexpr, n_block_size: tl.constexpr, k_block_size: tl.constexpr,
    group_size_m: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m_size, m_block_size)
    num_pid_n = tl.cdiv(n_size, n_block_size)
    num_pid_in_group = group_size_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size_m
    group_size_m = min(num_pid_m - first_pid_m, group_size_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute offsets
    offs_am = (pid_m * m_block_size + tl.arange(0, m_block_size)) % m_size
    offs_bn = (pid_n * n_block_size + tl.arange(0, n_block_size)) % n_size
    offs_k = tl.arange(0, k_block_size)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((m_block_size, n_block_size), dtype=tl.float32)

    # Compute matrix multiplication in blocks
    for k in range(0, tl.cdiv(k_size, k_block_size)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_size - k * k_block_size, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_size - k * k_block_size, other=0.0)
        accumulator += tl.dot(a, b.to(tl.float16))
        a_ptrs += k_block_size * stride_ak
        b_ptrs += k_block_size * stride_bk
    
    c = accumulator

    # Store result in c_ptr
    offs_cm = pid_m * m_block_size + tl.arange(0, m_block_size)
    offs_cn = pid_n * n_block_size + tl.arange(0, n_block_size)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < m_size) & (offs_cn[None, :] < n_size)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b):
    # Ensure dimensions are valid for matrix multiplication
    assert a.shape[1] == b.shape[0], "Dimensions of matrices do not match for multiplication"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    m_size, k_size = a.shape
    k_size, n_size = b.shape

    # Allocate output matrix
    c = torch.empty((m_size, n_size), device=a.device, dtype=torch.float32)
    
    # Define grid for the kernel
    grid = lambda META: (triton.cdiv(m_size, META['m_block_size']) * triton.cdiv(n_size, META['n_block_size']),)

    # Call kernel to compute matrix multiplication
    matmul_kernel[grid](
        a, b, c,
        m_size, n_size, k_size,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        m_block_size=32, n_block_size=32, k_block_size=32,
        group_size_m=8
    )
    return c
