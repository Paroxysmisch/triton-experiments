import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 256,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_M': 256,
            'BLOCK_N': 128,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_M': 256,
            'BLOCK_N': 64,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 256,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 128,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 64,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 128,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 32,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 32,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        }, num_stages=5, num_warps=2),
    ],
    key=['CACHE_M', 'CACHE_N', 'CACHE_K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10
    },
)
@triton.jit
def log_softmax_kernel_matmul_fc(
    input_ptr,
    output_ptr,
    n_elements,
    stride_zx_m,
    stride_zx_n,
    CACHE_M: tl.constexpr,
    CACHE_N: tl.constexpr,
    CACHE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_blocks_m = tl.cdiv(CACHE_M, BLOCK_M)
    n_blocks_n = tl.cdiv(CACHE_N, BLOCK_N)
    n_blocks_in_group = n_blocks_m * n_blocks_n
    group_id = pid // n_blocks_in_group
    first_block_m = pid % n_blocks_m
    first_block_n = (pid % n_blocks_in_group) // n_blocks_m
    offset_block_m = first_block_m * BLOCK_M
    offset_block_n = first_block_n * BLOCK_N
    rpid = tl.arange(0, BLOCK_M) + offset_block_m
    cppid = tl.arange(0, BLOCK_N) + offset_block_n
    input_block_ptr = (input_ptr +
                       rpid[:, None] * stride_zx_m +
                       cppid[None, :] * stride_zx_n)
    group_size = BLOCK_K * SPLIT_K
    offset_group_id = group_id * group_size
    idx_j = (tl.arange(0, BLOCK_K) + offset_group_id) % CACHE_N
    input_block_ptr = input_block_ptr + idx_j[None, :] * stride_zx_n
    block_ptr = input_block_ptr
    limit_m = min(CACHE_M, (group_id + 1) * group_size)
    limit_n = min(CACHE_N, (group_id + 1) * group_size)
    z = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, CACHE_K, BLOCK_K):
        idx_k = tl.arange(0, BLOCK_K) + k
        x = tl.load(
            block_ptr,
            mask=((None)[:, None] >= idx_k[None, :]) &
            (idx_k[:, None] >= (None)[None, :]),
            other=float("-inf"),
        )
        z = tl.where((None)[:, None] >= idx_k[None, :], x, z)
        block_ptr += BLOCK_K * stride_zx_k
    z = tl.where((None)[:, None] >= limit_n, z, float("-inf"))
    for i in range(0, BLOCK_M, BLOCK_N):
        idx_i = tl.arange(0, BLOCK_N) + i
        m = tl.load(z, mask=(None)[:, None] >= idx_i[None, :], other=float("-inf"))
        m_minus_max = m - tl.max(m, axis=1)[:, None]
        numerator = tl.exp(m_minus_max)
        denominator = tl.sum(numerator, axis=1)[:, None]
        softmax = numerator / denominator
        log_softmax = m_minus_max - tl.log(denominator)
        idx_i_clamped = tl.minimum(idx_i, limit_m - 1)
        rpid = tl.arange(0, BLOCK_N) + i
        output_block_ptr = (output_ptr +
                            rpid[:, None] * stride_zx_m +
                            idx_j[None, :] * stride_zx_n)
        tl.store(
            output_block_ptr,
            log_softmax,
            mask=(None)[:, None] >= idx_i[None, :],
        )
        idx_j_clamped = tl.minimum(idx_j, limit_n - 1)
        tl.store(
            output_block_ptr,
            log_softmax,
            mask=idx_j_clamped[None, :] >= (None)[:, None],
        )

def fused_cross_entropy_log_softmax(
    input: torch.Tensor,
    target: torch.Tensor,
    dim: int = 1,
    weight: torch.Tensor = None,
