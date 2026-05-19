import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'ROWS_PER_PROGRAM': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'ROWS_PER_PROGRAM': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 512, 'ROWS_PER_PROGRAM': 2}, num_warps=8),
    ],
    key=['M', 'N']
)
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    # Inputs
    X, Y, W, B,
    # Optional inputs
    RESIDUAL, X1, W1, B1,
    # Params
    M, N,
    # Normalization params
    eps: tl.constexpr, use_rms_norm: tl.constexpr,
    # Dropout params
    p: tl.constexpr, SEEDS, DROPOUT_MASK,
    # Mode flags
    has_residual: tl.constexpr, has_rowscale: tl.constexpr,
    has_X1: tl.constexpr, residual_in_fp32: tl.constexpr,
    # Outputs
    Mean, Rstd, RESIDUAL_OUT,
    # Optional outputs
    Y1,
    # Blocking
    BLOCK_SIZE_N: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr,
):
    row_block = tl.program_id(0)
    row_start = row_block * ROWS_PER_PROGRAM
    
    # Offset into outputs
    y_offsets = (row_start + tl.arange(0, ROWS_PER_PROGRAM))[:, None] * N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    mask = (row_start + tl.arange(0, ROWS_PER_PROGRAM))[:, None] < M
    
    # Load input data
    x_ptrs = X + y_offsets
    x = tl.load(x_ptrs, mask=mask, other=0)
    
    if has_rowscale:
        rowscale = tl.load(ROWSCALE + row_start + tl.arange(0, ROWS_PER_PROGRAM)[:, None], mask=mask[:, 0], other=1.0)
        x = x * rowscale

    # Compute mean and variance
    if not use_rms_norm:
        mean = tl.sum(x, axis=1) / N
        x_zm = x - mean[:, None]
        var = tl.sum(x_zm * x_zm, axis=1) / N
    else:
        var = tl.sum(x * x, axis=1) / N
        x_zm = x
    
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store statistics if needed
    if not use_rms_norm:
        mean_ptrs = Mean + (row_start + tl.arange(0, ROWS_PER_PROGRAM))
        tl.store(mean_ptrs, mean, mask=mask[:, 0])
    
    rstd_ptrs = Rstd + (row_start + tl.arange(0, ROWS_PER_PROGRAM))
    tl.store(rstd_ptrs, rstd, mask=mask[:, 0])

    # Normalize
    w = tl.load(W + tl.arange(0, BLOCK_SIZE_N), mask=tl.arange(0, BLOCK_SIZE_N) < N)
    y = x_zm * rstd[:, None] * w[None, :]
    
    # Add bias if present
    if B is not None:
        b = tl.load(B + tl.arange(0, BLOCK_SIZE_N), mask=tl.arange(0, BLOCK_SIZE_N) < N)
        y += b[None, :]

    # Apply dropout
    if p > 0.0:
        seed = tl.load(SEEDS + row_start + tl.arange(0, ROWS_PER_PROGRAM))
        random = tl.rand(seed, tl.arange(0, BLOCK_SIZE_N))
        dropout_mask = random > p
        y = tl.where(dropout_mask, y / (1.0 - p), 0.0)
        tl.store(DROPOUT_MASK + y_offsets, dropout_mask, mask=mask)

    # Handle residual connection
    if has_residual:
        residual = tl.load(RESIDUAL + y_offsets, mask=mask, other=0)
        if residual_in_fp32:
            residual = residual.to(tl.float32)
        y += residual

        if RESIDUAL_OUT is not None:
            tl.store(RESIDUAL_OUT + y_offsets, y, mask=mask)

    # Process X1 if present
    if has_X1:
        x1 = tl.load(X1 + y_offsets, mask=mask, other=0)
        w1 = tl.load(W1 + tl.arange(0, BLOCK_SIZE_N), mask=tl.arange(0, BLOCK_SIZE_N) < N)
        y1_val = x1 * w1[None, :]
        if B1 is not None:
            b1 = tl.load(B1 + tl.arange(0, BLOCK_SIZE_N), mask=tl.arange(0, BLOCK_SIZE_N) < N)
            y1_val += b1[None, :]
        tl.store(Y1 + y_offsets, y1_val, mask=mask)

    tl.store(Y + y_offsets, y, mask=mask)

def layer_norm_1pass(
    X: torch.Tensor, W: torch.Tensor, B: torch.Tensor,
    residual: torch.Tensor = None,
    X1: torch.Tensor = None, W1: torch.Tensor = None, B1: torch.Tensor = None,
    eps: float = 1e-5, p: float = 0.0, rowscale: torch.Tensor = None,
    residual_out: torch.Tensor = None, return_dropout_mask: bool = False,
    use_rms_norm: bool = False,
):
    M, N = X.shape
    device = X.device
    
    # Allocate outputs
    Y = torch.empty_like(X)
    Y1 = torch.empty_like(X) if X1 is not None else None
    
    # Statistics buffers
    Mean = torch.empty(M, device=device) if not use_rms_norm else None
    Rstd = torch.empty(M, device=device)
    
    # Dropout setup
    if p > 0:
        seeds = torch.randint(0, 2**32, (M,), device=device)
        dropout_mask = torch.empty((M, N), device=device, dtype=torch.bool) if return_dropout_mask else None
    else:
        seeds = None
        dropout_mask = None

    # Grid configuration
    grid = lambda meta: (triton.cdiv(M, meta['ROWS_PER_PROGRAM']),)
    
    # Kernel execution
    _layer_norm_fwd_1pass_kernel[grid](
        X, Y, W, B,
        residual, X1, W1, B1,
        M, N,
        eps, use_rms_norm,
        p, seeds, dropout_mask,
        residual is not None, rowscale is not None,
        X1 is not None, residual.dtype == torch.float32 if residual is not None else False,
        Mean, Rstd, residual_out,
        Y1,
        BLOCK_SIZE_N=min(triton.next_power_of_2(N), 512),
        ROWS_PER_PROGRAM=4,
    )
    
    outputs = (Y, Mean, Rstd) if not use_rms_norm else (Y, Rstd)
    if Y1 is not None:
        outputs += (Y1,)
    if return_dropout_mask:
        outputs += (dropout_mask,)
        
    return outputs

# Basic usage
x = torch.randn(1024, 512, device='cuda')
w = torch.ones(512, device='cuda')
b = torch.zeros(512, device='cuda')

y, mean, rstd = layer_norm_1pass(x, w, b)

# With dropout and residual
residual = torch.randn_like(x)
y, mean, rstd, mask = layer_norm_1pass(
    x, w, b, residual=residual,
    p=0.1, return_dropout_mask=True
)

# With secondary input processing
x1 = torch.randn_like(x)
w1 = torch.ones_like(w)
y, mean, rstd, y1 = layer_norm_1pass(
    x, w, b, X1=x1, W1=w1
)
