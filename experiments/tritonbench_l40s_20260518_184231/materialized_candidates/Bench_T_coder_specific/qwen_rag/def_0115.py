@triton.jit
def kernel_fwd(
    C,  # Pointer to output tensor
    A,  # Pointer to input tensor
    B,  # Pointer to weight matrix
    bias,  # Pointer to bias vector
    # Matrix dimensions
    M,
    N,
    K,
    # Stride variables
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(A + rm[:, None] * stride_am + rk[None, :] * stride_ak)
        b = tl.load(B + rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        acc += tl.dot(a, b)

    if bias is not None:
        bias = tl.load(bias + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    C = C + rm[:, None] * stride_am + rn[None, :]
    tl.store(C, tl.tanh(acc))
