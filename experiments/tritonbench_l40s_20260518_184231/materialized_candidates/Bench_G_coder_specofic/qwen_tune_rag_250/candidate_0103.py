# number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    rows_per_program: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_EVEN_N: tl.constexpr
):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Offset data pointers to the start of the row
    X += row_start * stride_x_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row

    # Load data to SRAM
    if IS_EVEN_N:
        x = tl.load(X + cols)
        dy = tl.load(DY + cols, mask=mask)
        w = tl.load(W + cols, mask=mask)
    else:
        x = tl.load(X + cols, mask=mask)
        dy = tl.load(DY + cols, mask=mask)
        w = tl.load(W + cols, mask=mask)

    # Compute dx
    rstd = tl.load(Rstd + row_start)
    x_hat = x * rstd
    dw = tl.zeros((rows_per_program, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(N, BLOCK_N)):
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask=mask)
        dy = tl.load(DY + cols, mask=mask)

        w = tl.load(W + cols, mask=mask)
        x_hat = x * rstd
        dw += dy * x_hat
        X += BLOCK_N
        DY += BLOCK_N
        W += BLOCK_N

    dw = tl.sum(dw, axis=0)
    if IS_EVEN_N:
        tl.store(DW + cols, dw)
    else:
        tl.store(DW + cols, dw, mask=mask)

def rmsnorm_triton_fwd(
    X: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not X.is_contiguous():
        X = X.contiguous()

    M, N = X.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    IS_EVEN_N = BLOCK_SIZE == N
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    BLOCK_SIZE = 256 * num_warps

    Rstd = torch.empty((M,), dtype=torch.float32, device=X.device)
    Y = torch.empty_like(X)
    grid = (M,)

    _rmsnorm_fwd_kernel[grid](
        X,
        Y,
        weight,
        Rstd,
        X.stride(0),
        Y.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_SIZE,
        num_warps=num_warps,
        IS_EVEN_N=IS_EVEN_N
    )

    return Y, Rstd

def rmsnorm_triton_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not dy.is_contiguous():
        dy = dy.contiguous()

    M, N = dy.shape
    rows_per_program = 128
    num_programs = (M + rows_per_program - 1) // rows_per_program
    BLOCK_SIZE = triton.next_power_of_2(N)
    IS_EVEN_N = BLOCK_SIZE == N
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    grid = (num_programs,)

    dw = torch.empty_like(weight)
    dx = torch.empty_like(x)

    _rmsnorm_bwd_kernel[grid](
        x,
        weight,
        dy,
        dx,
        dw,
        rstd,
        x.stride(0),
        dy.stride(0),
        dx.stride(0),
        M,
        N,
        eps,
        rows_per_program=rows_per_program,
        BLOCK_N=BLOCK_SIZE,
        num_warps=num_warps,
        IS_EVEN_N=IS_EVEN_N
    )

    return dx, dw
