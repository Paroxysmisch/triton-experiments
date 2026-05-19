B += BLOCK_K * SPLIT_K * stride_bk
        offs_k += BLOCK_K * SPLIT_K

    acc = acc.to(C.dtype.element_ty)

    offs_cm = offs_m + offs_am
    offs_cn = offs_n + offs_bn

    C = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = get_2d_mask(offs_cm, offs_cn, M, N)

    tl.store(C, acc, mask=mask)

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
@triton.jit()
def matmul_kernel_grouped_splitk_col_major(
    A, B, C, 
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    acc_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr
):
    pid, pid_z = tl.program_id(0), tl.program_id(1)

    pid_m, pid_n = col_major(pid, M, N, BLOCK_M, BLOCK_N)

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
        offs_k += BLOCK_K * SPLIT_K

    acc = acc.to(C.dtype.element_ty)

    offs_cm = offs_m + offs_am
    offs_cn = offs_n + offs_bn

    C = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = get_2d_mask(offs_cm, offs_cn, M, N)

    tl.store(C, acc, mask=mask)

@triton.jit()
def matmul_kernel_grouped(
        A, B, C, 
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        acc_dtype: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        AB_DTYPE: tl.constexpr,
        EVEN_K: tl.constexpr
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = get_1d_offset(BLOCK_M, pid_m)
    offs_n = get_1d_offset(BLOCK_N, pid_n)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_amk = get_2d_offset(offs_am, offs_k, stride_0=stride_am, stride_1=stride_ak)
    offs_bkn = get_2d_offset(offs_k, offs_bn, stride_0=stride_bk, stride_1=stride_bn)

    A = A + offs_amk
    B = B + offs_bkn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * BLOCK_K
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=offs_k[:, None] < k_remaining, other=_0)
            b = tl.load(B, mask=offs_k[None, :] < k_remaining, other=_0)

        if AB_DTYPE is not None:
            a = a.to(AB_DTYPE)
            b = b.to(AB_DTYPE)

        acc += tl.dot(a, b, out_dtype=acc_dtype)

        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    acc = acc.to(C.dtype.element_ty)

    offs_cm = offs_m + offs_am
    offs_cn = offs_n + offs_bn

    C = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = get_2d_mask(offs_cm, offs_cn, M, N)

    tl.store(C, acc, mask=mask)

@triton.autotune(
    configs=[
        Config(meta={'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        Config(meta={'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        Config(meta={'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        Config(meta={'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        Config(meta={'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        Config(meta={'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        Config(meta={'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16, 'SPLIT_K': 1, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        Config(meta={'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16, 'SPLIT_K': 1, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        Config(meta={'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16, 'SPLIT_K': 1, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
