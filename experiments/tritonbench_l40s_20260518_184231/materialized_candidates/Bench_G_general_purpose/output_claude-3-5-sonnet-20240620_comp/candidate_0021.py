import torch
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A, B, O,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_ob, stride_om, stride_on,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr, DIVISIBLE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, TILE_M)
    num_pid_n = tl.cdiv(N, TILE_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * TILE_M + tl.arange(0, TILE_M)) % M
    offs_bn = (pid_n * TILE_N + tl.arange(0, TILE_N)) % N
    offs_k = tl.arange(0, TILE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    
    for k in range(0, K, TILE_K):
        if DIVISIBLE_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            mask_k = k + offs_k < K
            a = tl.load(a_ptrs, mask=mask_k[None, :])
            b = tl.load(b_ptrs, mask=mask_k[:, None])
        accumulator += tl.dot(a, b)
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

    offs_om = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_on = pid_n * TILE_N + tl.arange(0, TILE_N)
    o_ptrs = O + (offs_om[:, None] * stride_om + offs_on[None, :] * stride_on)
    
    if DIVISIBLE_M and DIVISIBLE_N:
        tl.store(o_ptrs, accumulator)
    else:
        mask_m = offs_om < M
        mask_n = offs_on < N
        tl.store(o_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])

@triton.autotune(
    configs=[
        triton.Config({'TILE_M': 64, 'TILE_N': 64, 'TILE_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'TILE_M': 128, 'TILE_N': 128, 'TILE_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'TILE_M': 64, 'TILE_N': 128, 'TILE_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'TILE_M': 128, 'TILE_N': 64, 'TILE_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bmm(A, B, O, M, N, K, batch_size):
    TILE_M, TILE_N, TILE_K, GROUP_M = tl.constexpr(64), tl.constexpr(64), tl.constexpr(32), tl.constexpr(8)
    DIVISIBLE_M = tl.constexpr(M % TILE_M == 0)
    DIVISIBLE_N = tl.constexpr(N % TILE_N == 0)
    DIVISIBLE_K = tl.constexpr(K % TILE_K == 0)

    num_pid_m = tl.cdiv(M, TILE_M)
    num_pid_n = tl.cdiv(N, TILE_N)
    num_pid_in_group = GROUP_M * num_pid_n
    grid = (num_pid_m * num_pid_n * batch_size,)

    bmm_kernel[grid](
        A, B, O,
        M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        TILE_M, TILE_N, TILE_K,
        GROUP_M,
        DIVISIBLE_M, DIVISIBLE_N, DIVISIBLE_K,
    )

def batched_matrix_multiply(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dim() == B.dim() == 3, "Input tensors must be 3-dimensional"
    assert A.shape[0] == B.shape[0], "Batch sizes must match"
    assert A.shape[2] == B.shape[1], "Inner dimensions must match"
    
    batch_size, M, K = A.shape
    _, K, N = B.shape
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Initialize output tensor
    O = torch.empty((batch_size, M, N), device=A.device, dtype=A.dtype)
    
    # Launch Triton kernel
    bmm(A, B, O, M, N, K, batch_size)
    
    return O

# Example usage
if __name__ == "__main__":
    batch_size, M, N, K = 32, 128, 64, 256
    A = torch.randn((batch_size, M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((batch_size, K, N), device='cuda', dtype=torch.float32)
    
    # Run Triton implementation
    O_triton = batched_matrix_multiply(A, B)
    
    # Run PyTorch implementation for comparison
    O_torch = torch.bmm(A, B)
    
    # Check results
    assert torch.allclose(O_triton, O_torch, atol=1e-2, rtol=1e-2)
    print("Triton implementation matches PyTorch!")
