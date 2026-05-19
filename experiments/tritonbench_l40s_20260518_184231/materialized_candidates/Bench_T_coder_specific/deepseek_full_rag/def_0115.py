lu(acc)

    # rm (resp. rn) denotes a range of indices for rows (resp. cols) of C
    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # C = C + cm[:, None] * stride_cm + cn[None, :] * stride_cn
    C = C + cm[:, None] * stride_cm + cn[None, :] * stride_cn
    # mask = (cm < M)[:, None] & (cn < N)[None, :]
    # tl.store(C, acc, mask=mask)
    tl.store(C, acc)

@triton.jit
def kernel_bwd(
    ACT_INPUT,
    D,
    D_bar,
    A,
    B,
    bias,
    # matrix dimensions
    M,
    N,
    K,
    # stride variables
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    # meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    A_ROWMAJOR: tl.constexpr,
    B_COLMAJOR: tl.constexpr,
    BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """
    Kernel for computing dA = dOut * activation_grad(A x W + C) x W^T
    - Input has shape (M, K)
    - Weight has shape (K, N)
    - Bias has shape (N,)
    - D_bar has shape (M, N)
    - Output has shape (M, K)
    'ActInputs' has shape (M, N) and saves the A x W + C intermediate
    This kernel will consolidate over K
    """
    pid = tl.program_id(axis=0)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # now compute the block that each program will go through
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # trick to avoid masking on M and N axis
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    if A_ROWMAJOR:
        A = A + (ram[:, None] * stride_am + rk[None, :])
    else:
        A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    if B_COLMAJOR:
        B = B + (rk[:, None] + rbn[None, :] * stride_bn)
    else:
        B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    # initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    # load D_bar into SRAM, compute D = D_bar * activation_grad(A x W + C)
    # then do dot product with W^T to get the final dx
    # this is to save L2 bandwidth
    # we can't load D_bar and D in the same loop because
    # (1) it's too big (~16kB per program)
    # (2) we have no idea where it is being cached in L2
    # (3) NVIDIA claims SRAM is faster than L2
    # hence, we load D_bar then do the dot product with W^T
    # we load D (the gradient) the same way

    # load D_bar into SRAM, compute D = D_bar * activation_grad(A x W + C)
    if ACTIVATION == "gelu":
        d_bar = tl.load(D_bar + ram[:, None] * stride_cm + rbn[None, :] * stride_cn)
        d_bar = gelu_grad(d_bar)
    elif ACTIVATION == "gelu_approx":
        d_bar = tl.load(D_bar + ram[:, None] * stride_cm + rbn[None, :] * stride_cn)
        d_bar = gelu_approx_grad(d_bar)
    elif ACTIVATION == "squared_relu":
        d_bar = tl.load(D_bar + ram[:, None] * stride_cm + rbn[None, :] * stride_cn)
        d_bar = squared_relu_grad(d_bar)

    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(d_bar, b.to(tl.float32), allow_tf32=False)

        if A_ROWMAJOR:
            A += BLOCK_K
        else:
            A += BLOCK_K * stride_ak
        if B_COLMAJOR:
            B += BLOCK_K
        else:
            B += BLOCK_K * stride_bk

    # mask = (ram < M)[:, None] & (rbn < N)[None, :]
    # d = tl.load(D, mask=mask)
    d = tl.load(D)
    # mask = (ram < M)[:, None] & (rk < K)[None, :]
    # tl.store(A, d[:, None] * acc, mask=mask)
    tl.store(A, d[:, None] * acc)

def triton_linear_act(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    act_input: Optional[torch.Tensor] = None,
    save_act_input: bool = False,
    activation: str = "none",
) -> torch.Tensor:
    """
    Argument:
        input: (M, K) tensor
        weight: (K, N) tensor
        bias: (N,) tensor
    Return:
        (M, N) tensor
    """
    assert input.dtype == weight.dtype
    assert input.is_contiguous()
    assert weight.is_contiguous()
    if bias is not None:
        assert bias.is_contiguous()
    if act_input is not None:
        assert act_input.is_contiguous()

    M, K = input.shape
    K, N = weight.shape
    assert K == N
    assert input.shape[1] == weight.
