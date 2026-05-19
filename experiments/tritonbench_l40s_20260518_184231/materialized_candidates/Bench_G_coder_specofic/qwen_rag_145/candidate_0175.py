@triton.jit
def _l2_norm_bwd_kernel(
    DX,  # pointer to the input gradient
    DY,  # pointer to the output gradient
    X,  # pointer to the input
    stride_x_row,  # stride for movement in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    DY += row * stride_x_row
    X += row * stride_x_row
    DX += row * stride_x_row

    _var = tl.zeros([BLOCK_N], dtype=tl.float32)  # Local summation on each block
    for off in range(0, N, BLOCK_N):
        cols = tl.arange(off, off + BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _var += x * x

    var = tl.sum(_var) / N  # Global summation on each row
    dx = tl.zeros([N], dtype=tl.float32)

    if tl.program_id(0) == 0:  # Only one block per row may calculate rstd & dx
        rstd = 1 / tl.sqrt(var + eps)
        dy = tl.load(DY, mask=cols < N).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dx = dy * rstd - tl.sum(dy * x) * (1 / (var + eps)) * rstd * x
        tl.store(DX + cols, dx, mask=cols < N)

def _l2_norm_bwd(
    DX,  # output: input gradient
    DY,  # input/output: output gradient
    X,  # input: input data
    eps=1e-5,  # epsilon for numerical stability
):
    X = X.reshape(-1, X.shape[-1])
    DY = DY.reshape(-1, DY.shape[-1])
    M, N = X.shape
    DX = torch.empty_like(X)  # Initialize output tensor
    stride_x_row = X.stride(0)
    MAX_FUSED_SIZE = 65536 // X.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    _l2_norm_bwd_kernel[(M,), BLOCK_N](
        DX,
        DY,
        X,
        stride_x_row,
        N,
        eps,
    )
    return DX.reshape(Dx_shape)  # Reshape back to original shape
