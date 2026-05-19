import torch
import triton
import triton.language as tl

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
    # Get program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute offsets
    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    y_ptrs = y_ptr + (offs_k[:, None] * stride_yk + offs_yn[None, :] * stride_yn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Compute matrix multiplication in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        y = tl.load(y_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(x, y.to(tl.float16))
        x_ptrs += BLOCK_SIZE_K * stride_xk
        y_ptrs += BLOCK_SIZE_K * stride_yk
    
    z = accumulator

    # Store result in z_ptr
    offs_zm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_zn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    z_ptrs = z_ptr + stride_zm * offs_zm[:, None] + stride_zn * offs_zn[None, :]
    z_mask = (offs_zm[:, None] < M) & (offs_zn[None, :] < N)
    tl.store(z_ptrs, z, mask=z_mask)

def matmul(x, y):
    # Ensure dimensions are valid for matrix multiplication
    assert x.shape[1] == y.shape[0], "Dimensions of matrices do not match for multiplication"
    assert x.is_contiguous(), "Matrix X must be contiguous"
    assert y.is_contiguous(), "Matrix Y must be contiguous"
    
    M, K = x.shape
    K, N = y.shape

    # Allocate output matrix
    z = torch.empty((M, N), device=x.device, dtype=torch.float32)
    
    # Define grid for the kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    # Call kernel to compute matrix multiplication
    matmul_kernel[grid](
        x, y, z,
        M, N, K,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        z.stride(0), z.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,  # Example block sizes
        GROUP_SIZE_M=8  # Example group size
    )
    return z
