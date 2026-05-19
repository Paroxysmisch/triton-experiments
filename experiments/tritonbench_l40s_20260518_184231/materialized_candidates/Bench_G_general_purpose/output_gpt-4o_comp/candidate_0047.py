import torch
import triton
import triton.language as tl

# Define constants
MAX_FUSED_SIZE = 1024

def calculate_settings(n):
    block_size = min(n, MAX_FUSED_SIZE)
    num_warps = min(8, max(1, block_size // 256))
    return block_size, num_warps

@triton.jit
def layernorm_forward(X, W, b, R, MU, Out, n_cols, BLOCK_SIZE: tl.constexpr):
    # Calculate row index
    row_idx = tl.program_id(0)
    # Calculate start index of the row
    row_start = row_idx * n_cols

    # Load data for the row
    x = tl.load(X + row_start + tl.arange(0, BLOCK_SIZE))
    
    # Calculate mean
    mu = tl.sum(x, axis=0) / n_cols
    # Calculate variance
    r = tl.sqrt(tl.sum((x - mu) ** 2, axis=0) / n_cols + 1e-5)

    # Store intermediate results
    tl.store(R + row_idx, r)
    tl.store(MU + row_idx, mu)

    # Normalize
    x_hat = (x - mu) / r
    # Scale and shift
    y = x_hat * tl.load(W + tl.arange(0, BLOCK_SIZE)) + tl.load(b + tl.arange(0, BLOCK_SIZE))
    # Store result
    tl.store(Out + row_start + tl.arange(0, BLOCK_SIZE), y)

@triton.jit
def layernorm_backward(dY, X, W, R, MU, dX, n_cols, BLOCK_SIZE: tl.constexpr):
    # Calculate row index
    row_idx = tl.program_id(0)
    # Calculate start index of the row
    row_start = row_idx * n_cols

    # Load data for the row
    dy = tl.load(dY + row_start + tl.arange(0, BLOCK_SIZE))
    x = tl.load(X + row_start + tl.arange(0, BLOCK_SIZE))
    r = tl.load(R + row_idx)
    mu = tl.load(MU + row_idx)

    # Calculate gradients
    x_hat = (x - mu) / r
    dx_hat = dy * tl.load(W + tl.arange(0, BLOCK_SIZE))
    
    dvar = tl.sum(dx_hat * (x - mu) * -0.5 * r**-3, axis=0)
    dmu = tl.sum(dx_hat * -1.0 / r, axis=0) + dvar * tl.sum(-2.0 * (x - mu) / n_cols, axis=0)

    dx = dx_hat / r + dvar * 2.0 * (x - mu) / n_cols + dmu / n_cols
    # Store gradient
    tl.store(dX + row_start + tl.arange(0, BLOCK_SIZE), dx)

class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b):
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        # Allocate memory for intermediate results
        R = torch.empty(n_rows, dtype=X.dtype, device=X.device)
        MU = torch.empty(n_rows, dtype=X.dtype, device=X.device)
        Out = torch.empty_like(X)

        # Launch forward kernel
        layernorm_forward[(n_rows,)](X, W, b, R, MU, Out, n_cols, BLOCK_SIZE=BLOCK_SIZE)

        # Save context for backward pass
        ctx.save_for_backward(X, W, R, MU)
        ctx.BLOCK_SIZE = BLOCK_SIZE

        return Out

    @staticmethod
    def backward(ctx, dY):
        X, W, R, MU = ctx.saved_tensors
        n_rows, n_cols = X.shape
        BLOCK_SIZE = ctx.BLOCK_SIZE

        # Allocate memory for gradient
        dX = torch.empty_like(X)

        # Launch backward kernel
        layernorm_backward[(n_rows,)](dY, X, W, R, MU, dX, n_cols, BLOCK_SIZE=BLOCK_SIZE)

        return dX, None, None

def fast_layernorm(X, layernorm_module):
    W = layernorm_module.weight
    b = layernorm_module.bias
    return Fast_Layernorm.apply(X, W, b)

# Example usage
# Assume `layernorm` is a PyTorch nn.LayerNorm module
# X is the input tensor
# output = fast_layernorm(X, layernorm)
