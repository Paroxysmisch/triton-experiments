import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
from triton.runtime.dlrm.block_16816.kernel import pack_weight, MAX_FUSED_SIZE
from triton.runtime.dlrm import packed_w_size

@triton.autotune(configs=kernel_utils.autotune_configs(), key=['M', 'N', 'K'], prune_configs_by={
    0: early_config_prune,
    'too_small_tile_m': estimate_matmul_time
}, merge_stages=True)
@triton.jit
def matmul_kernel(a_ptr: tl.tensor, b_ptr: tl.tensor, c_ptr: tl.tensor,
                  M: int, N: int, K: int,  # Matrix dimensions
                  stride_am: int, stride_ak: int,  # stride for a
                  stride_bk: int, stride_bn: int,  # stride for b
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,  # Block dimensions
                  BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                  next_block: tl.constexpr):  # block for ordering
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    block_offset_m = pid_m * BLOCK_SIZE_M
    block_offset_n = pid_n * BLOCK_SIZE_N
    offs_am = block_offset_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = block_offset_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.)
        if next_block:
            padding = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.)  # Load B[k*BLOCK_SIZE_K:(k+1)*BLOCK_SIZE_K, :]
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    tl.store(c_ptrs, c)


def matmul(a: torch.Tensor, b: torch.Tensor, next_block: bool = False) -> torch.Tensor:
    """wrapper around matmul_kernel"""
    assert a.shape[1] % 4 == 0, f'Matmul error: Matrix A should be of shape (M, 4*K), but was {a.shape}'
    assert a.is_contiguous(), 'Matmul error: tensor A must be contiguous'
    assert b.is_contiguous(), 'Matmul error: tensor B must be contiguous'

    k = a.shape[1] // 4
    N = b.shape[1]
    M = a.shape[0]
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    matmul_kernel[grid](a, b, c,  # tensor a, b, c
                        M, N, k,  # matrix dimensions
                        a.stride(0), a.stride(1),  # stride(a)
                        b.stride(0), b.stride(1),  # stride(b)
                        next_block=next_block)
    return c
