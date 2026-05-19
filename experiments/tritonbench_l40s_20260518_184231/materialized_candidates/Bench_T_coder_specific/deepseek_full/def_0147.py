import torch
import triton
import triton.language as tl

@triton.jit
def pairwise_distance_normalize_kernel(
    x1_ptr, x2_ptr, output_ptr,
    M, N, K,
    p_norm, eps_norm: tl.constexpr, eps_distance: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    offs_k = tl.arange(0, BLOCK_N)
    x1_ptrs = x1_ptr + offs_m[:, None] * N * K + offs_k[None, :] * K
    x2_ptrs = x2_ptr + offs_n[None, :] * K
    x1 = tl.load(x1_ptrs, mask=offs_k[None, :] < K, other=0.0)
    x2 = tl.load(x2_ptrs, mask=offs_k[:, None] < K, other=0.0)

    x1_norm = tl.sum(tl.abs(x1) ** p_norm, axis=1)
    x1_norm = tl.maximum(x1_norm, eps_norm)
    x1 = x1 / x1_norm[:, None]

    x2_norm = tl.sum(tl.abs(x2) ** p_norm, axis=1)
    x2_norm = tl.maximum(x2_norm, eps_norm)
    x2 = x2 / x2_norm[:, None]

    x1 = tl.where(offs_k[None, :] < K, x1, 0.0)
    x2 = tl.where(offs_k[:, None] < K, x2, 0.0)

    x1 = tl.sum(x1 * x2, axis=1)
    x1 = tl.maximum(x1, eps_distance)
    output = tl.sqrt(x1)

    offs = pid_m * N + pid_n
    output_ptr_mask = offs < M * N
    tl.store(output_ptr + offs, output, mask=output_ptr_mask)

def fused_pairwise_distance_normalize(
    x1: torch.Tensor, x2: torch.Tensor, p_norm: float = 2.0, eps_norm: float = 1e-12, eps_distance: float = 1e-6, keepdim: bool = False
) -> torch.Tensor:
    assert x1.dtype == x2.dtype
    assert x1.dtype in [torch.float16, torch.bfloat16, torch.float32]
    x1 = x1.contiguous()
    x2 = x2.contiguous()

    M, N, K = x1.shape[0], x2.shape[0], x1.shape[1]
    output = torch.empty((M, N), dtype=x1.dtype, device=x1.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    pairwise_distance_normalize_kernel[grid](x1, x2, output, M, N, K, p_norm, eps_norm, eps_distance)
    return output
