import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    x_ptr, y_ptr, z_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_yk, stride_yn,
    stride_zm, stride_zn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Calculate offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over k dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load x and y sub-matrices
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k * BLOCK_SIZE_K + offs_k[None, :]) * stride_xk, mask=offs_m[:, None] < M, other=0.0)
        y = tl.load(y_ptr + (k * BLOCK_SIZE_K + offs_k[:, None]) * stride_yk + offs_n[None, :] * stride_yn, mask=offs_n[None, :] < N, other=0.0)
        
        # Compute matrix multiplication
        acc += tl.dot(x, y)
    
    # Store result
    z = acc.to(tl.float16)
    tl.store(z_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn, z, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape[1] == y.shape[0], "Incompatible matrix dimensions"
    M, K = x.shape
    K, N = y.shape
    
    # Initialize output
    z = torch.empty((M, N), device=x.device, dtype=torch.float16)
    
    # Configure kernel parameters
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Launch kernel
    grid = lambda META: (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    matmul_kernel[grid](
        x, y, z,
        M, N, K,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        z.stride(0), z.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return z

# Example usage
if __name__ == "__main__":
    M, N, K = 1024, 1024, 1024
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    y = torch.randn(K, N, device="cuda", dtype=torch.float16)
    
    z_triton = matmul(x, y)
    z_torch = torch.matmul(x, y)
    
    print(f"Max difference: {torch.max(torch.abs(z_triton - z_torch))}")
