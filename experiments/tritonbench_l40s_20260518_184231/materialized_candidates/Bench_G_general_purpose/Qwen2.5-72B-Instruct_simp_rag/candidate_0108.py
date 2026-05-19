import triton
import triton.language as tl
import torch
import math
from typing import Tuple

# Forward kernel
@triton.jit
def _rmsnorm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    IS_EVEN_N: tl.constexpr
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row

    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    mask = cols < N
    if IS_EVEN_N:
        w = tl.load(W + cols).to(tl.float32)
    else:
        w = tl.load(W + cols, mask=mask).to(tl.float32)

    x_hat = x * rstd
    y = x_hat * w

    # Write output
    if IS_EVEN_N:
        tl.store(Y + cols, y)
    else:
        tl.store(Y + cols, y, mask=mask)

# Backward kernel
@triton.jit
def _rmsnorm_bwd_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DW,  # pointer to the partial sum of weights gradient
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_dy_row,
    stride_dx_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    rows_per_program,
    BLOCK_N: tl.constexpr,
    IS_EVEN_N: tl.constexpr
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row

    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row

    w = tl.load(W + cols, mask=mask).to(tl.float32)

    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)

    row_end = min((row_block_id + 1) * rows_per_program, M)

    for row in range(row_start, row_end):
        # Load data to SRAM
        if IS_EVEN_N:
            x = tl.load(X + cols).to(tl.float32)
            dy = tl.load(DY + cols).to(tl.float32)
        else:
            x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)

        rstd = tl.load(Rstd + row)

        # Compute dx
        xhat = x * rstd
        if not IS_EVEN_N:
            xhat = tl.where(mask, xhat, 0.0)

        wdy = w * dy
        dw += dy * xhat

        c1 = tl.sum(xhat * wdy, axis=0) / N
        dx = (wdy - xhat * c1) * rstd

        tl.store(DX + cols, dx, mask=mask)

        X += stride_x_row
        DY += stride_dy_row
        DX += stride_dx_row

    tl.store(DW + row_block_id * N + cols, dw, mask=mask)

# PyTorch autograd function for forward pass
@torch.library.custom_op("flasht5::rmsnorm_triton_fwd", mutates_args=(), device_types="cuda")
def rmsnorm_triton_fwd(
    X: torch.Tensor,
    weight: torch.Tensor,
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N = X.shape

    assert X.stride(-1) == 1

    assert weight.shape == (N,)
    assert weight.stride(-1) == 1

    # allocate output
    Y = torch.empty_like(X)
    assert Y.stride(-1) == 1

    rstd = torch.empty((M,), dtype=torch.float32, device=X.device)

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // X.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    assert N <= BLOCK_N

    # heuristics for number of warps
    with torch.cuda.device(X.device.index):
        _rmsnorm_fwd_kernel[(M,)](
            X,
            Y,
            weight,
            rstd,
            X.stride(0),
            Y.stride(0),
            N,
            eps,
            BLOCK_N,
            (N % BLOCK_N == 0)
        )

    return Y, rstd

# PyTorch autograd function for backward pass
@torch.library.custom_op("flasht5::rmsnorm_triton_bwd", mutates_args=(), device_types="cuda")
def rmsnorm_triton_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dy.shape == (M, N)

    assert weight.shape == (N,)
    assert weight.stride(-1) == 1

    # allocate output
    dx = torch.empty_like(x)

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    assert N <= BLOCK_N

    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)

    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    with torch.cuda.device(x.device.index):
        _rmsnorm_bwd_kernel[grid](
            x,
            weight,
            dy,
            dx,
            _dw,
            rstd,
            x.stride(0),
            dy.stride(0),
            dx.stride(0),
            M,
            N,
            eps,
            rows_per_program,
            BLOCK_N,
            (N % BLOCK_N == 0)
        )
    dw = _dw.sum(0).to(weight.dtype)

    return dx, dw

# Simple RMS LayerNorm module for testing
class RMSLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x):
        y, _ = rmsnorm_triton_fwd(x, self.weight, self.eps)
        return y

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    x = torch.randn(1024, 512, device="cuda")

    # Create a RMS LayerNorm layer
    rms_layer_norm = RMSLayerNorm(512).to("cuda")

    # Forward pass
    y = rms_layer_norm(x)

    # Compute loss (for example, mean squared error)
    loss = y.mean()

    # Backward pass
    loss.backward()

    print("Forward pass output shape:", y.shape)
    print("Gradient of the input:", rms_layer_norm.weight.grad)
