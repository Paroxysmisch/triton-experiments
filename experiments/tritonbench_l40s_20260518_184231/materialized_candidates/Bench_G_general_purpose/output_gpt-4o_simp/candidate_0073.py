import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2),
    ],
    key=['N']
)
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X, W, B, Y, M, V, N,  # Inputs and outputs
    eps,                  # Epsilon for numerical stability
    residual, rms_norm,   # Flags for optional features
    stride_xm, stride_ym, stride_n,  # Strides for input and output tensors
    BLOCK_SIZE: tl.constexpr  # Block size for parallelization
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input and compute mean
    x = tl.load(X + offsets * stride_n, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N

    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N

    # Normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * inv_std

    # Apply weights and biases
    if rms_norm:
        norm = tl.sqrt(tl.sum(x * x, axis=0) / N)
        x_norm = x / norm

    y = x_norm * tl.load(W + offsets * stride_n, mask=mask, other=1.0) + tl.load(B + offsets * stride_n, mask=mask, other=0.0)

    if residual:
        y += x

    # Store output
    tl.store(Y + offsets * stride_n, y, mask=mask)
    tl.store(M + offsets * stride_n, mean, mask=mask)
    tl.store(V + offsets * stride_n, var, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2),
    ],
    key=['N']
)
@triton.jit
def _layer_norm_bwd_kernel(
    DY, X, W, B, DX, DW, DB, M, V, N,  # Inputs and outputs
    eps,                              # Epsilon for numerical stability
    residual, rms_norm,               # Flags for optional features
    stride_dym, stride_xm, stride_n,  # Strides for input and output tensors
    BLOCK_SIZE: tl.constexpr  # Block size for parallelization
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load inputs and forward pass results
    dy = tl.load(DY + offsets * stride_n, mask=mask, other=0.0)
    x = tl.load(X + offsets * stride_n, mask=mask, other=0.0)
    mean = tl.load(M + offsets * stride_n, mask=mask, other=0.0)
    var = tl.load(V + offsets * stride_n, mask=mask, other=0.0)

    # Recompute forward output if needed
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_centered = x - mean
    x_norm = x_centered * inv_std

    if rms_norm:
        norm = tl.sqrt(tl.sum(x * x, axis=0) / N)
        x_norm = x / norm

    # Compute gradients
    dw = dy * x_norm
    db = dy
    dx_norm = dy * tl.load(W + offsets * stride_n, mask=mask, other=1.0)

    # Backprop through normalization
    dvar = tl.sum(dx_norm * x_centered * -0.5 * inv_std**3, axis=0)
    dmean = tl.sum(dx_norm * -inv_std, axis=0) + dvar * tl.sum(-2.0 * x_centered, axis=0) / N
    dx = dx_norm * inv_std + dvar * 2.0 * x_centered / N + dmean / N

    if residual:
        dx += dy

    # Store gradients
    tl.store(DX + offsets * stride_n, dx, mask=mask)
    tl.atomic_add(DW + offsets * stride_n, dw, mask=mask)
    tl.atomic_add(DB + offsets * stride_n, db, mask=mask)

def layer_norm(x, weight, bias, eps=1e-5, residual=False, rms_norm=False):
    # Determine the size of the input tensor
    N, M = x.shape

    # Allocate output tensors
    y = torch.empty_like(x)
    mean = torch.empty(N, dtype=x.dtype, device=x.device)
    var = torch.empty(N, dtype=x.dtype, device=x.device)

    # Launch the forward kernel
    _layer_norm_fwd_1pass_kernel[(N,)](
        x, weight, bias, y, mean, var, N,
        eps, residual, rms_norm,
        x.stride(0), y.stride(0), x.stride(1)
    )

    return y, mean, var

def layer_norm_backward(dy, x, weight, bias, mean, var, eps=1e-5, residual=False, rms_norm=False):
    # Determine the size of the input tensor
    N, M = x.shape

    # Allocate gradient tensors
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    db = torch.empty_like(bias)

    # Launch the backward kernel
    _layer_norm_bwd_kernel[(N,)](
        dy, x, weight, bias, dx, dw, db, mean, var, N,
        eps, residual, rms_norm,
        dy.stride(0), x.stride(0), x.stride(1)
    )

    return dx, dw, db
