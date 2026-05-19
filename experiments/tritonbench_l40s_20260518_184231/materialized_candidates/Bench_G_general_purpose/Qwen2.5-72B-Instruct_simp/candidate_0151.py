import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X, Y, W, B, 
    RESIDUAL, X1, W1, B1, 
    Y1, MEAN, RSTD, 
    RESIDUAL_OUT, SEEDS, DROPOUT_MASK, DROPOUT_MASK1, 
    N, E, EPS, DROPOUT_P, 
    IS_RMS_NORM: tl.constexpr, 
    HAS_RESIDUAL: tl.constexpr, 
    HAS_X1: tl.constexpr, 
    HAS_W1: tl.constexpr, 
    HAS_B1: tl.constexpr, 
    HAS_DROPOUT: tl.constexpr, 
    HAS_DROPOUT1: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_size = tl.num_programs(axis=0)
    row = pid * N

    # Load input data
    x_ptr = X + row
    w_ptr = W + row
    b_ptr = B + row
    x = tl.load(x_ptr, mask=row < N * E, other=0.0)
    w = tl.load(w_ptr, mask=row < N * E, other=0.0)
    b = tl.load(b_ptr, mask=row < N * E, other=0.0)

    # Compute mean and variance
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / E
        var = tl.sum((x - mean) ** 2, axis=0) / E
        rstd = 1.0 / tl.sqrt(var + EPS)
    else:
        mean = 0.0
        rstd = 1.0 / tl.sqrt(tl.sum(x ** 2, axis=0) / E + EPS)

    # Normalize
    x_norm = (x - mean) * rstd
    y = x_norm * w + b

    # Apply dropout if enabled
    if HAS_DROPOUT:
        seed = tl.load(SEEDS + pid)
        dropout_mask = tl.rand(seed, row) > DROPOUT_P
        y = tl.where(dropout_mask, y, 0.0)
        tl.store(DROPOUT_MASK + row, dropout_mask, mask=row < N * E)

    # Store the normalized output
    tl.store(Y + row, y, mask=row < N * E)

    # Compute and store mean and rstd
    if MEAN is not None:
        tl.store(MEAN + pid, mean)
    if RSTD is not None:
        tl.store(RSTD + pid, rstd)

    # Handle residual and additional tensors
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + row, mask=row < N * E, other=0.0)
        y = y + residual
        tl.store(RESIDUAL_OUT + row, y, mask=row < N * E)

    if HAS_X1:
        x1_ptr = X1 + row
        x1 = tl.load(x1_ptr, mask=row < N * E, other=0.0)
        x1_norm = (x1 - mean) * rstd

        if HAS_W1:
            w1_ptr = W1 + row
            w1 = tl.load(w1_ptr, mask=row < N * E, other=0.0)
            x1_norm = x1_norm * w1

        if HAS_B1:
            b1_ptr = B1 + row
            b1 = tl.load(b1_ptr, mask=row < N * E, other=0.0)
            x1_norm = x1_norm + b1

        y1 = x1_norm

        if HAS_DROPOUT1:
            seed1 = tl.load(SEEDS + pid + block_size)
            dropout_mask1 = tl.rand(seed1, row) > DROPOUT_P
            y1 = tl.where(dropout_mask1, y1, 0.0)
            tl.store(DROPOUT_MASK1 + row, dropout_mask1, mask=row < N * E)

        tl.store(Y1 + row, y1, mask=row < N * E)

import torch

def layer_norm_fwd_1pass(
    X, W, B, 
    RESIDUAL=None, X1=None, W1=None, B1=None, 
    Y=None, Y1=None, MEAN=None, RSTD=None, 
    RESIDUAL_OUT=None, SEEDS=None, DROPOUT_MASK=None, DROPOUT_MASK1=None, 
    EPS=1e-5, DROPOUT_P=0.0, IS_RMS_NORM=False, 
    HAS_RESIDUAL=False, HAS_X1=False, HAS_W1=False, HAS_B1=False, HAS_DROPOUT=False, HAS_DROPOUT1=False
):
    N, E = X.shape
    assert W.shape == (E,)
    assert B.shape == (E,)
    if RESIDUAL is not None:
        assert RESIDUAL.shape == (N, E)
    if X1 is not None:
        assert X1.shape == (N, E)
    if W1 is not None:
        assert W1.shape == (E,)
    if B1 is not None:
        assert B1.shape == (E,)
    if Y is None:
        Y = torch.empty_like(X)
    if Y1 is not None:
        assert Y1.shape == (N, E)
    if MEAN is not None:
        assert MEAN.shape == (N,)
    if RSTD is not None:
        assert RSTD.shape == (N,)
    if RESIDUAL_OUT is not None:
        assert RESIDUAL_OUT.shape == (N, E)
    if SEEDS is not None:
        assert SEEDS.shape == (N,)
    if DROPOUT_MASK is not None:
        assert DROPOUT_MASK.shape == (N, E)
    if DROPOUT_MASK1 is not None:
        assert DROPOUT_MASK1.shape == (N, E)

    grid = (N,)

    _layer_norm_fwd_1pass_kernel[grid](
        X, Y, W, B, 
        RESIDUAL, X1, W1, B1, 
        Y1, MEAN, RSTD, 
        RESIDUAL_OUT, SEEDS, DROPOUT_MASK, DROPOUT_MASK1, 
        N, E, EPS, DROPOUT_P, 
        IS_RMS_NORM, 
        HAS_RESIDUAL, 
        HAS_X1, 
        HAS_W1, 
        HAS_B1, 
        HAS_DROPOUT, 
        HAS_DROPOUT1
    )

    return Y, Y1, MEAN, RSTD, RESIDUAL_OUT, DROPOUT_MASK, DROPOUT_MASK1
