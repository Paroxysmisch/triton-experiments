import torch
import triton
import triton.language as tl
from triton import Config
from triton_util import get_1d_offset, get_2d_offset, get_1d_mask, get_2d_mask
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

def get_configs_io_bound(do_split_k=False, do_col_major=False):
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    
                    if do_split_k:
                        for split_k in [2, 4, 8]:
                            configs.append(
                                Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k, 'GROUP_SIZE_M': 8},
                                    num_stages=num_stages, num_warps=num_warps, pre_hook=lambda nargs: nargs['C'].zero_()))
                    elif do_col_major:
                        configs.append(
                        Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1},
                               num_stages=num_stages, num_warps=num_warps))
                    else:
                        configs.append(
                        Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1, 'GROUP_SIZE_M': 8},
                               num_stages=num_stages, num_warps=num_warps))
    return configs                    

@triton.jit()
def col_major(pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    pid_m = (pid % grid_m)
    pid_n = pid // grid_m

    return pid_m, pid_n

@triton.autotune(
    configs=get_configs_io_bound(do_split_k=True),
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10,
    },
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def matmul_kernel_grouped_splitk(
        A, B, C, 
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        acc_dtype: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        SPLIT_K: tl.constexpr,
        EVEN_K: tl.constexpr,
        AB_DTYPE: tl.constexpr
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m_t, pid_n_t = pid // num_pid_n, pid % num_pid_n 

    pid_m, pid_n = tl.swizzle2d(pid_m_t, pid_n_t, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_m = get_1d_offset(BLOCK_M, pid_m)
    offs_n = get_1d_offset(BLOCK_N, pid_n)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)
    offs_k = get_1d_offset(BLOCK_K, pid_z)

    offs_amk = get_2d_offset(offs_am, offs_k, stride_0=stride_am, stride_1=stride_ak)
    offs_bkn = get_2d_offset(offs_k, offs_bn, stride_0=stride_bk, stride_1=stride_bn)

    A = A + offs_amk
    B = B + offs_bkn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=offs_k[:, None] < k_remaining, other=_0)
            b = tl.load(B, mask=offs_k[None, :] < k_remaining, other=_0)

        if AB_DTYPE is not None:
            a = a.to(AB_DTYPE)
            b = b.to(AB_DTYPE)

        acc += tl.dot(a, b, out_dtype=acc_dtype)

        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    acc = acc.to(C.type.element_ty)

    offs_m = get_1d_offset(BLOCK_M, pid_m)
    offs_n = get_1d_offset(BLOCK_N, pid_n)

    offs_cmn = get_2d_offset(offs_m, offs_n, stride_cm, stride_cn)
    
    C = C + offs_cmn
    mask = get_2d_mask(offs_m, offs_n, M, N)

    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)

@triton.autotune(
    configs=get_configs_io_bound(do_split_k=False),
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10,
    },
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def matmul_kernel_grouped(
        A, B, C, 
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        acc_dtype: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        SPLIT_K: tl.constexpr,
        EVEN_K: tl.constexpr,
        AB_DTYPE: tl.constexpr
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m_t, pid_n_t = pid // num_pid_n, pid % num_pid_n 

    pid_m, pid_n = tl.swizzle2d(pid_m_t, pid_n_t, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_m = get_1d_offset(BLOCK_M, pid_m)
    offs_n = get_1d_offset(BLOCK_N, pid_n)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)
    offs_k = get_1d_offset(BLOCK_K, pid_z)

    offs_amk = get_2d_offset(offs_am, offs_k, stride_0=stride_am, stride_1=stride_ak)
    offs_bkn = get_2d_offset(offs_k, offs_bn, stride_0=stride_bk, stride_1=stride_bn)

    A = A + offs_amk
    B = B + offs_bkn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=offs_k[:, None] < k_remaining, other=_0)
            b = tl.load(B, mask=offs_k[None, :] < k_remaining, other=_0)

        if AB_DTYPE is not None:
            a = a.to(AB_DTYPE)
            b = b.to(AB_DTYPE)

        acc += tl.dot(a, b, out_dtype=acc_dtype)

        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    acc = acc.to(C.type.element_ty
