import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import num_warps_range, partial_unroll

@triton.jit
def matmul4_kernel(a_ptr, qb_ptr, scale_ptr, zero_point_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn,
                   stride_cm, stride_cn, stride_scale, stride_zero_point,
                   BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                   BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am
    qb_ptrs = qb_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    scale_ptrs = scale_ptr + offs_bn[None, :]
    zero_point_ptrs = zero_point_ptr + offs_bn[None, :]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # modify here
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)

        offs_bk = offs_k[:, None] + k * BLOCK_SIZE_K
        # modify here
        quantized_b_block = tl.load(qb_ptrs + offs_bk * stride_bk, mask=offs_bk < K,
                                    other=0)  # doesn't fit in BLOCK_SIZE_K
        scale = tl.load(scale_ptrs + offs_bk * stride_bk, mask=offs_bk < K, other=0.0)
        zero_point = tl.load(zero_point_ptrs + offs_bk * stride_bk, mask=offs_bk < K,
                             other=0)  # doesn't fit in BLOCK_SIZE_K

        # constrain to BLOCK_SIZE_K
        quantized_b_block = tl.where(offs_bk[None, :] < K,
                                     quantized_b_block, 0)  # out of bounds is zero
        scale = tl.where(offs_bk[None, :] < K, scale,
                         0.0)  # out of bounds is zero-float so not problematic
        zero_point = tl.where(offs_bk[None, :] < K, zero_point,
                              0)  # out of bounds is zero

        # dequantize_int4
        b = (quantized_b_block - zero_point) * scale  # broadcast
        accumulator += tl.dot(a, b)

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (stride_cm * offs_cm[:, None] +
                      stride_cn * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

tuningtime_list = []
@triton.autotune(configs=_get_configs(matmul4_kernel), key=['M', 'N', 'K'], prune_configs_by={
    'success': lambda x: (1e-4, 1e-2)})
@partial_unroll(VERTICAL=1, M=2)
@num_warps_range(1, None)
def matmul_dequantize_int4_gptq(a, qb, scale, zero_point, out=None,
                                a_rowmajor: bool = False, inplace: bool = False):
    if a_rowmajor:
        assert a.stride(0) == 1 and qb.stride(0) == 1
    else:
        assert a.stride(-1) == 1 and qb.stride(-1) == 1

    M, N = a.shape
    K, N = qb.shape
    assert a.shape[1] == qb.shape[1], "incompatible dimensions"
    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    elif inplace:
        assert out.shape == (M, N)
        c = out
    else:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        tmp, N = torch._inductor.runtime.optimization_context.get_tmp_space(c)
        c = out

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M'])
                         * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    start = time.time()
    matmul4_kernel[grid](
        a,
        qb,
        scale,
        zero_point,
        c,
        M,
        N,
        K,
        a.stride(0) if a_rowmajor else a.stride(-1),
        a.stride(1) if a_rowmajor else a.stride(0),
        qb.stride(0),
        qb.stride(1),
        c.stride(0),
        c.stride(1),
    )
    end = time.time()
    tuning_time = end - start
    tuningtime_list.append(tuning_time)
    return c
