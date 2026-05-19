import torch
import triton
import triton.language as tl

@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
    "HAS_Z": lambda args: args["Z"] is not None,
    "HAS_DROPOUT": lambda args: args["dropout_p"] > 0.0,
    "HAS_RESIDUAL": lambda args: args["RESIDUAL"] is not None
})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X, Y, W, B, Z, RESIDUAL,  # Input tensors
    Mean, Rstd, SEEDS, DROPOUT_MASK,  # Output tensors
    stride_x_row, stride_y_row, stride_z_row, stride_residual_row, stride_dropout_row,
    M, N, eps, dropout_p,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_DROPOUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    
    # Adjust pointers for group and row
    X += row * stride_x_row + group * N
    Y += row * stride_y_row + group * N
    if HAS_Z:
        Z += row * stride_z_row + group * N
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_residual_row + group * N
    if HAS_DROPOUT:
        DROPOUT_MASK += row * stride_dropout_row + group * N
        SEEDS += group * M + row
        
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)

    # Gating before normalization
    if HAS_Z and not NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask).to(tl.float32)
        x = x * z * tl.sigmoid(z)

    # Compute mean and variance
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = x - mean
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # Normalize and transform
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    y = x_hat * w + b if HAS_BIAS else x_hat * w

    # Gating after normalization
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask).to(tl.float32)
        y = y * z * tl.sigmoid(z)

    # Apply dropout
    if HAS_DROPOUT:
        seed = tl.load(SEEDS)
        random = tl.rand(seed, cols)
        keep = random >= dropout_p
        y = tl.where(keep, y / (1 - dropout_p), 0.0)
        tl.store(DROPOUT_MASK + cols, keep, mask=mask)

    # Add residual
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=mask).to(tl.float32)
        y += residual

    tl.store(Y + cols, y, mask=mask)

def layer_norm_forward(
    x, weight, bias, z=None, residual=None,
    eps=1e-5, dropout_p=0.0, group_size=None,
    norm_before_gate=True, is_rms_norm=False
):
    M, N = x.shape
    group_size = group_size or N
    assert N % group_size == 0, "Feature dim must be divisible by group size"
    ngroups = N // group_size
    
    # Allocate output tensors
    y = torch.empty_like(x)
    mean = torch.empty(M*ngroups, device=x.device, dtype=torch.float32) if not is_rms_norm else None
    rstd = torch.empty(M*ngroups, device=x.device, dtype=torch.float32)
    
    # Dropout setup
    dropout_mask, seeds = None, None
    if dropout_p > 0:
        dropout_mask = torch.empty_like(x, dtype=torch.bool)
        seeds = torch.randint(0, 2**32, (M*ngroups,), device=x.device, dtype=torch.int32)
    
    # Kernel configuration
    BLOCK_N = triton.next_power_of_2(group_size)
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M, ngroups)
    
    # Launch kernel
    _layer_norm_fwd_1pass_kernel[grid](
        x, y, weight, bias, z, residual, mean, rstd, seeds, dropout_mask,
        x.stride(0), y.stride(0), 
        z.stride(0) if z is not None else 0,
        residual.stride(0) if residual is not None else 0,
        dropout_mask.stride(0) if dropout_mask is not None else 0,
        M, group_size, eps, dropout_p,
        BLOCK_N=BLOCK_N,
        HAS_BIAS=bias is not None,
        HAS_Z=z is not None,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        HAS_DROPOUT=dropout_p > 0,
        HAS_RESIDUAL=residual is not None,
        num_warps=num_warps
    )
    
    return y, mean, rstd, dropout_mask

# Example usage
x = torch.randn(1024, 512, device='cuda')
weight = torch.ones(512, device='cuda')
bias = torch.zeros(512, device='cuda')

# With dropout and residual
output, mean, rstd, mask = layer_norm_forward(
    x, weight, bias,
    dropout_p=0.2,
    residual=x.clone()
)
