import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_fused(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride_ml,
    stride_n,
    L,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Implements the forward pass of LayerNorm.
    
    Parameters:
    X (tl.tensor): Input tensor.
    Y (tl.tensor): Output tensor for normalized features.
    W (tl.tensor): Weights for scaling the normalized data.
    B (tl.tensor): Biases for shifting the normalized data.
    Mean (tl.tensor): Tensor to store the mean of each row.
    Rstd (tl.tensor): Tensor to store the reciprocal of the standard deviation of each row.
    stride_ml (int): Stride to access elements along the combined dimensions M and L.
    stride_n (int): Stride to access elements along dimension N.
    L (int): Size of the second dimension in the batch.
    N (int): Total number of features per instance.
    eps (float): Small epsilon value for numerical stability in division.
    BLOCK_SIZE (tl.constexpr): Block size used for partitioning computations.
    """
    # Setup for batched execution over M and L
    row = tl.program_id(0)
    batch = tl.program_id(1)

    # Calculate the base index for the current matrix slice
    base_idx = row * stride_ml + batch * stride_n
    Y += base_idx
    X += base_idx

    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sum_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _sum += a
        _sum_squares += a * a

    mean = tl.sum(_sum) / N
    var = tl.sum(_sum_squares) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Store the mean and reciprocal of the standard deviation
    tl.store(Mean + row * L + batch, mean)
    tl.store(Rstd + row * L + batch, rstd)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def _layer_norm_bwd_dx_fused(
    X,
    W,
    B,
    DY,
    DX,
    DW,
    DB,
    Mean,
    Rstd,
    stride_ml,
    stride_n,
    L,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Implements the backward pass of LayerNorm to compute gradients with respect to the input and weights.
    
    Parameters:
    X (tl.tensor): Input tensor.
    W (tl.tensor): Weights for scaling the normalized data.
    B (tl.tensor): Biases for shifting the normalized data.
    DY (tl.tensor): Gradient of the output tensor.
    DX (tl.tensor): Gradient of the input tensor.
    DW (tl.tensor): Gradient of the weights.
    DB (tl.tensor): Gradient of the biases.
    Mean (tl.tensor): Mean of each row.
    Rstd (tl.tensor): Reciprocal of the standard deviation of each row.
    stride_ml (int): Stride to access elements along the combined dimensions M and L.
    stride_n (int): Stride to access elements along dimension N.
    L (int): Size of the second dimension in the batch.
    N (int): Total number of features per instance.
    eps (float): Small epsilon value for numerical stability in division.
    BLOCK_SIZE (tl.constexpr): Block size used for partitioning computations.
    """
    # Setup for batched execution over M and L
    row = tl.program_id(0)
    batch = tl.program_id(1)

    # Calculate the base index for the current matrix slice
    base_idx = row * stride_ml + batch * stride_n
    X += base_idx
    DY += base_idx
    DX += base_idx

    mean = tl.load(Mean + row * L + batch)
    rstd = tl.load(Rstd + row * L + batch)

    _sum_dy = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sum_dy_x_hat = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        _sum_dy += dy
        _sum_dy_x_hat += dy * x_hat

    sum_dy = tl.sum(_sum_dy)
    sum_dy_x_hat = tl.sum(_sum_dy_x_hat)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        x_hat = (x - mean) * rstd
        dx = dy * w * rstd - (sum_dy * w * rstd + sum_dy_x_hat * x_hat * rstd * rstd * rstd) / N
        tl.store(DX + cols, dx, mask=mask)

        # Accumulate gradients for weights and biases
        tl.atomic_add(DB + cols, dy, mask=mask)
        tl.atomic_add(DW + cols, dy * x_hat, mask=mask)

@triton.jit
def _layer_norm_bwd_dwdb(
    DW,
    DB,
    FINAL_DW,
    FINAL_DB,
    L,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Aggregates partial weight and bias gradients to produce the final gradients.
    
    Parameters:
    DW (tl.tensor): Partial gradients of the weights.
    DB (tl.tensor): Partial gradients of the biases.
    FINAL_DW (tl.tensor): Final gradients of the weights.
    FINAL_DB (tl.tensor): Final gradients of the biases.
    L (int): Size of the second dimension in the batch.
    N (int): Total number of features per instance.
    BLOCK_SIZE (tl.constexpr): Block size used for partitioning computations.
    """
    # Setup for batched execution over M and L
    row = tl.program_id(0)
    batch = tl.program_id(1)

    # Calculate the base index for the current matrix slice
    base_idx = row * L + batch

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        dw = tl.load(DW + base_idx * N + cols, mask=mask, other=0.0)
        db = tl.load(DB + base_idx * N + cols, mask=mask, other=0.0)
        tl.atomic_add(FINAL_DW + cols, dw, mask=mask)
        tl.atomic_add(FINAL_DB + cols, db, mask=mask)
