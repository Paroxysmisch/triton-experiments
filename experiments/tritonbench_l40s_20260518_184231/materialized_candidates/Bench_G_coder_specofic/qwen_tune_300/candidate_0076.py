import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'num_stages': 1, 'num_warps': 8}),
        triton.Config({'num_stages': 2, 'num_warps': 8}),
        triton.Config({'num_stages': 4, 'num_warps': 8}),
        triton.Config({'num_stages': 8, 'num_warps': 8}),
        triton.Config({'num_stages': 1}),
        triton.Config({'num_stages': 2}),
        triton.Config({'num_stages': 4}),
        triton.Config({'num_stages': 8}),
        triton.Config({'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def quantize_int8_perrow_kernel(
    fpa,
    a,
    as_,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    max_ = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    for k in range(0, K, BLOCK_SIZE_K):
        a_ptrs = fpa + offs_m[:, None] * K + k + offs_k[None, :]
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        a_val = tl.load(a_ptrs, mask=a_mask, other=0.0)
        max_ = tl.maximum(max_, tl.max(tl.abs(a_val), axis=1))
    scale = tl.math.nextafter(max_ * 127.0 / 255.0, 0.0)

    for k in range(0, K, BLOCK_SIZE_K):
        a_ptrs = fpa + offs_m[:, None] * K + k + offs_k[None, :]
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        a_val = tl.load(a_ptrs, mask=a_mask, other=0.0)
        int_val = (a_val / scale[:, None]).to(tl.int8)
        a_ptrs = a + offs_m[:, None] * N + offs_n[None, :] + k + offs_k[None, :]
        tl.store(a_ptrs, int_val, mask=(offs_n[None, :] < N) & ((k + offs_k[None, :]) < K))
    as_1 = tl.broadcast_to(scale[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    as_ptrs = as_ + offs_m[:, None] * N + offs_n[None, :]
    tl.store(as_ptrs, as_1, mask=(offs_n[None, :] < N))


def quantize_int8_perrow(fpa):
    M, K = fpa.shape
    a = torch.empty((M, N), device=fpa.device, dtype=torch.int8)
    as_ = torch.empty((M, N), device=fpa.device, dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    quantize_int8_perrow_kernel[grid](fpa, a, as_, M, N, K)
    return a, as_


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1, 'num_stages': 2, 'num_warps': 16}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'SPLIT_K': 1, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'SPLIT_K': 1, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'SPLIT_K': 1, 'num_stages': 2, 'num_warps': 16}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'SPLIT_K': 1, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'SPLIT_K': 1, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'SPLIT_K': 1, 'num_stages': 2, 'num_warps': 16}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 2, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'SPLIT_K': 2, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'SPLIT_K': 2, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 4, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'SPLIT_K': 4, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'SPLIT_K': 4, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'SPLIT_K': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 16, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a,
    b,
    c,
    as_ptr,
    bs_ptr,
    groupsize,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE
