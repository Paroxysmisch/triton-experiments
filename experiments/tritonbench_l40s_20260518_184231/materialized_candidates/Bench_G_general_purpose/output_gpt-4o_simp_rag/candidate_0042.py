import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X, Y, stride_x, stride_y, N, eps, BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for L2 normalization forward pass.
    
    Parameters:
    X (tl.tensor): Input tensor.
    Y (tl.tensor): Output tensor for normalized data.
    stride_x (int): Stride for input tensor.
    stride_y (int): Stride for output tensor.
    N (int): Number of features per row.
    eps (float): Small epsilon value for numerical stability.
    BLOCK_SIZE (tl.constexpr): Block size for computation.
    """
    row_idx = tl.program_id(0)
    base_idx = row_idx * stride_x
    X += base_idx
    Y += base_idx

    # Initialize sum of squares
    sum_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        sum_squares += x * x

    norm_factor = tl.sqrt(tl.sum(sum_squares) + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        y = x / norm_factor
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX, stride_x, stride_dy, stride_dx, N, eps, BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for L2 normalization backward pass.
    
    Parameters:
    X (tl.tensor): Input tensor.
    DY (tl.tensor): Gradient of the output tensor.
    DX (tl.tensor): Gradient of the input tensor.
    stride_x (int): Stride for input tensor.
    stride_dy (int): Stride for output gradient tensor.
    stride_dx (int): Stride for input gradient tensor.
    N (int): Number of features per row.
    eps (float): Small epsilon value for numerical stability.
    BLOCK_SIZE (tl.constexpr): Block size for computation.
    """
    row_idx = tl.program_id(0)
    base_idx_x = row_idx * stride_x
    base_idx_dy = row_idx * stride_dy
    base_idx_dx = row_idx * stride_dx

    X += base_idx_x
    DY += base_idx_dy
    DX += base_idx_dx

    # Initialize sum of squares and dot product
    sum_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    dot_product = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=cols < N, other=0.0).to(tl.float32)
        sum_squares += x * x
        dot_product += x * dy

    norm_factor = tl.sqrt(tl.sum(sum_squares) + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        dx = (dy - (dot_product / (norm_factor * norm_factor * norm_factor)) * x) / norm_factor
        tl.store(DX + cols, dx, mask=mask)

def _l2_norm_fwd(X, Y, N, eps=1e-5, BLOCK_SIZE=128):
    """
    Wrapper for L2 normalization forward pass.
    
    Parameters:
    X (torch.Tensor): Input tensor.
    Y (torch.Tensor): Output tensor for normalized data.
    N (int): Number of features per row.
    eps (float): Small epsilon value for numerical stability.
    BLOCK_SIZE (int): Block size for computation.
    """
    grid = (X.shape[0],)
    stride_x = X.stride(0)
    stride_y = Y.stride(0)
    triton.launch(
        _l2_norm_fwd_1pass_kernel,
        grid=grid,
        num_warps=4,
        num_stages=2,
        X=X,
        Y=Y,
        stride_x=stride_x,
        stride_y=stride_y,
        N=N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def _l2_norm_bwd(X, DY, DX, N, eps=1e-5, BLOCK_SIZE=128):
    """
    Wrapper for L2 normalization backward pass.
    
    Parameters:
    X (torch.Tensor): Input tensor.
    DY (torch.Tensor): Gradient of the output tensor.
    DX (torch.Tensor): Gradient of the input tensor.
    N (int): Number of features per row.
    eps (float): Small epsilon value for numerical stability.
    BLOCK_SIZE (int): Block size for computation.
    """
    grid = (X.shape[0],)
    stride_x = X.stride(0)
    stride_dy = DY.stride(0)
    stride_dx = DX.stride(0)
    triton.launch(
        _l2_norm_bwd_kernel,
        grid=grid,
        num_warps=4,
        num_stages=2,
        X=X,
        DY=DY,
        DX=DX,
        stride_x=stride_x,
        stride_dy=stride_dy,
        stride_dx=stride_dx,
        N=N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
