import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

@triton.jit
def matmul_kernel(a, b, c,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr,
                  num_stages: tl.constexpr,
                  num_warps: tl.constexpr):
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

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    for _ in range(num_stages):
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            a_vals = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
            b_vals = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
            accumulator += tl.dot(a_vals, b_vals)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        a_ptrs += BLOCK_SIZE_M * stride_am
        b_ptrs += BLOCK_SIZE_N * stride_bn
        c_ptrs = c + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
        accumulator = tl.where(c_mask, accumulator, 0.0)
        tl.store(c_ptrs, accumulator, mask=c_mask)

def matmul(a, b, a_batch_dim, a_batch_stride, a_M, a_K,
           b_batch_dim, b_batch_stride, b_K, b_N,
           c_batch_dim, c_batch_stride, c_M, c_N,
           config=None,
           stream=None,
           index_dtype=None,
           dtype=None,
           grid=None,
           num_warps=None,
           num_stages=None,
           acc_dtype=None):
    # check constraints
    assert a_K == b_K, "Incompatible dimensions"
    assert a_M >= 0 and a_N >= 0 and b_M >= 0 and b_N >= 0, "Negative dimension is not allowed"
    assert a_batch_dim >= 0 and a_batch_dim < a.dim(), "Invalid a_batch_dim"
    assert b_batch_dim >= 0 and b_batch_dim < b.dim(), "Invalid b_batch_dim"
    assert a_batch_stride > 0 and b_batch_stride > 0 and c_batch_stride > 0, "Invalid batch stride"

    if config is None:
        if dtype is None:
            dtype = a.dtype
        if index_dtype is None:
            index_dtype = torch.int64
        if num_warps is None:
            num_warps = 4
        if num_stages is None:
            num_stages = 2
        if acc_dtype is None:
            acc_dtype = dtype
        config = GEMV_CONFIGS[dtype][index_dtype][num_warps]

    config = early_config_prune(config, a_K, num_stages, num_warps)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = config["block_size"]
    GROUP_SIZE_M = config["group_size"]
    num_warps = config["num_warps"]

    grid = (triton.cdiv(a_M, BLOCK_SIZE_M) * triton.cdiv(b_N, BLOCK_SIZE_N), )

    if stream is None:
        stream = torch.cuda.current_stream(0)

    a_stride_m = a.stride(a_batch_dim + 1) if a_batch_dim + 1 < a.ndim else 0
    a_stride_k = a.stride(a_batch_dim + 2) if a_batch_dim + 2 < a.ndim else 0
    b_stride_k = b.stride(b_batch_dim + 1) if b_batch_dim + 1 < b.ndim else 0
    b_stride_n = b.stride(b_batch_dim + 2) if b_batch_dim + 2 < b.ndim else 0
    c_stride_m = c_batch_stride if c_batch_stride > 0 else c.stride(c_batch_dim + 1) if c_batch_dim + 1 < c.ndim else 0
    c_stride_n = c_batch_stride if c_batch_stride > 0 else c.stride(c_batch_dim + 2) if c_batch_dim + 2 < c.ndim else 0

    matmul_kernel[grid](
        a, b, c,
        a_M, a_N, a_K,
        a_stride_m, a_stride_k,
        b_stride_k, b_stride_n,
        c_stride_m, c_stride_n,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_stages=num_stages,
        num_warps=num_warps,
        stream=stream,
    )

    return estimate_matmul_time(a_M, a_N, a_K, config)
