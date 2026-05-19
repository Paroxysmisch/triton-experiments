import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_forward_kernel(
    X, mean, var, W, b, r, mu, n,
    stride0, stride1,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr = 1e-5
):
    pid = tl.program_id(0)
    X_ptr = X + pid * stride0
    mean_ptr = mean + pid
    var_ptr = var + pid
    W_ptr = W + pid
    b_ptr = b + pid
    r_ptr = r + pid
    mu_ptr = mu + pid

    # Compute mean and variance
    x_sum = 0.0
    x2_sum = 0.0
    for i in range(BLOCK_SIZE):
        x = tl.load(X_ptr + i * stride1)
        x_sum += x
        x2_sum += x * x
    mean_val = x_sum / BLOCK_SIZE
    var_val = x2_sum / BLOCK_SIZE - mean_val * mean_val + EPS

    # Store mean and variance
    tl.store(mean_ptr, mean_val)
    tl.store(var_ptr, var_val)

    # Normalize and scale/shift
    for i in range(BLOCK_SIZE):
        x = tl.load(X_ptr + i * stride1)
        x_normalized = (x - mean_val) / tl.sqrt(var_val)
        x_scaled = x_normalized * tl.load(W_ptr) + tl.load(b_ptr)
        tl.store(r_ptr + i * stride1, x_scaled)

@triton.jit
def layernorm_backward_kernel(
    dY, X, r, mu, var, W, dX, n,
    stride0, stride1,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr = 1e-5
):
    pid = tl.program_id(0)
    dY_ptr = dY + pid * stride0
    X_ptr = X + pid * stride0
    r_ptr = r + pid
    mu_ptr = mu + pid
    var_ptr = var + pid
    W_ptr = W + pid
    dX_ptr = dX + pid * stride0

    # Compute dX
    dY_sum = 0.0
    for i in range(BLOCK_SIZE):
        dY_val = tl.load(dY_ptr + i * stride1)
        dX_val = tl.load(dX_ptr + i * stride1)
        dY_sum += dY_val

    for i in range(BLOCK_SIZE):
        x = tl.load(X_ptr + i * stride1)
        x_normalized = (x - tl.load(mu_ptr)) / tl.sqrt(tl.load(var_ptr))
        dY_val = tl.load(dY_ptr + i * stride1)
        dX_val = dY_val / tl.sqrt(tl.load(var_ptr))
        dX_val -= (x_normalized * dY_sum) / BLOCK_SIZE
        dX_val -= (x_normalized * dY_val) / BLOCK_SIZE
        dX_val += (2.0 * (x - tl.load(mu_ptr)) * dY_val) / (BLOCK_SIZE * tl.sqrt(tl.load(var_ptr)))
        tl.store(dX_ptr + i * stride1, dX_val)

class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b):
        n = X.shape[1]
        stride0 = X.stride(0)
        stride1 = X.stride(1)
        BLOCK_SIZE = min(256, n)

        mean = torch.zeros(X.shape[0], device=X.device)
        var = torch.zeros(X.shape[0], device=X.device)
        r = torch.zeros_like(X)
        mu = torch.zeros_like(X)

        layernorm_forward_kernel[grid=(X.shape[0],), block=(BLOCK_SIZE,)](
            X, mean, var, W, b, r, mu, n,
            stride0, stride1,
            BLOCK_SIZE
        )

        ctx.save_for_backward(X, mean, var, W, b, r, mu)
        return r

    @staticmethod
    def backward(ctx, dY):
        X, mean, var, W, b, r, mu = ctx.saved_tensors
        n = X.shape[1]
        stride0 = X.stride(0)
        stride1 = X.stride(1)
        BLOCK_SIZE = min(256, n)

        dX = torch.zeros_like(X)
        layernorm_backward_kernel[grid=(X.shape[0],), block=(BLOCK_SIZE,)](
            dY, X, r, mu, var, W, dX, n,
            stride0, stride1,
            BLOCK_SIZE
        )

        return dX, None, None

def fast_layernorm(X, layer_norm):
    W = layer_norm.weight
    b = layer_norm.bias
    return Fast_Layernorm.apply(X, W, b)
