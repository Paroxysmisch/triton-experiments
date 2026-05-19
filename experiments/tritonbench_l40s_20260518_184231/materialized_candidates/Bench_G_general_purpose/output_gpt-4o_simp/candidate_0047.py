import torch
import triton
import triton.language as tl

# Kernel for forward pass of layer normalization
@triton.jit
def layernorm_forward(X_ptr, gamma_ptr, beta_ptr, Y_ptr, mean_ptr, rstd_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get the program id
    pid = tl.program_id(0)
    # Create a block of indices
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to prevent out-of-bounds memory access
    mask = offsets < N
    # Load input, gamma, and beta
    X = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
    # Compute mean
    mean = tl.sum(X, axis=0) / N
    tl.store(mean_ptr + pid, mean)
    # Compute variance and rstd (reciprocal of standard deviation)
    var = tl.sum((X - mean) * (X - mean), axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(rstd_ptr + pid, rstd)
    # Normalize and apply gamma and beta
    Y = (X - mean) * rstd * gamma + beta
    tl.store(Y_ptr + offsets, Y, mask=mask)

# Kernel for backward pass of layer normalization
@triton.jit
def layernorm_backward(dY_ptr, X_ptr, mean_ptr, rstd_ptr, gamma_ptr, dX_ptr, dgamma_ptr, dbeta_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    # Load data
    dY = tl.load(dY_ptr + offsets, mask=mask, other=0.0)
    X = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + pid)
    rstd = tl.load(rstd_ptr + pid)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    # Compute gradients
    X_hat = (X - mean) * rstd
    dX_hat = dY * gamma
    dvar = tl.sum(dX_hat * (X - mean) * (-0.5) * rstd**3, axis=0)
    dmean = tl.sum(dX_hat * (-rstd), axis=0) + dvar * tl.sum(-2.0 * (X - mean), axis=0) / N
    dX = dX_hat * rstd + dvar * 2.0 * (X - mean) / N + dmean / N
    dgamma = tl.sum(dY * X_hat, axis=0)
    dbeta = tl.sum(dY, axis=0)
    # Store gradients
    tl.store(dX_ptr + offsets, dX, mask=mask)
    tl.store(dgamma_ptr + offsets, dgamma, mask=mask)
    tl.store(dbeta_ptr + offsets, dbeta, mask=mask)

# Function to calculate block size and number of warps
def calculate_settings(num_columns):
    BLOCK_SIZE = 128
    num_warps = 4
    if num_columns <= 128:
        BLOCK_SIZE = 128
        num_warps = 4
    elif num_columns <= 256:
        BLOCK_SIZE = 256
        num_warps = 8
    else:
        BLOCK_SIZE = 512
        num_warps = 16
    return BLOCK_SIZE, num_warps

# PyTorch custom autograd function
class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, gamma, beta):
        # Determine settings
        N = X.shape[1]
        BLOCK_SIZE, num_warps = calculate_settings(N)
        # Allocate output tensors
        Y = torch.empty_like(X)
        mean = torch.empty(X.shape[0], device=X.device)
        rstd = torch.empty(X.shape[0], device=X.device)
        # Launch forward kernel
        layernorm_forward[(X.shape[0],)](X, gamma, beta, Y, mean, rstd, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        # Save context for backward pass
        ctx.save_for_backward(X, gamma, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return Y

    @staticmethod
    def backward(ctx, dY):
        X, gamma, mean, rstd = ctx.saved_tensors
        BLOCK_SIZE = ctx.BLOCK_SIZE
        N = X.shape[1]
        # Allocate gradients
        dX = torch.empty_like(X)
        dgamma = torch.empty_like(gamma)
        dbeta = torch.empty_like(gamma)
        # Launch backward kernel
        layernorm_backward[(X.shape[0],)](dY, X, mean, rstd, gamma, dX, dgamma, dbeta, N, BLOCK_SIZE=BLOCK_SIZE)
        return dX, dgamma, dbeta

# Utility function to apply fast layer normalization
def fast_layernorm(X, gamma, beta):
    return Fast_Layernorm.apply(X, gamma, beta)

# Example usage
if __name__ == "__main__":
    X = torch.randn(64, 256, device='cuda')
    gamma = torch.ones(256, device='cuda')
    beta = torch.zeros(256, device='cuda')
    Y = fast_layernorm(X, gamma, beta)
    print(Y)
