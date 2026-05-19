import torch
import triton
import triton.language as tl

@triton.jit
def _rms_layernorm_forward(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    R,  # pointer to the row-wise variances
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr = 4,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    r = tl.math.rsqrt(var)
    tl.store(R + row, r)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        y = x * r * w
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def _rms_layernorm_backward(
    X,  # pointer to the input
    DX,  # pointer to input's residual
    DW,  # pointer to weight's residual
    Y,  # pointer to the output
    W,  # pointer to the weights
    R,  # pointer to the row-wise variances
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr = 4,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    DY = Y  # read Y as DY for computation convenience
    DX += row * stride
    # Compute the gradients
    _dy = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _dw = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        y = tl.load(Y + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        v = x * tl.load(R + row)
        _dy += dy * v
        _dw += dy * w * v
    # Store DX and DW
    r = tl.load(R + row)
    dw = tl.sum(_dw, axis=0) * r / N
    tl.store(DW + row, dw)
    dy = tl.sum(_dy, axis=0) / N
    tl.store(DX + row, dy)

@triton.jit
def _gemma_rms_layernorm_forward(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    R,  # pointer to the row-wise variances
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr = 4,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    r = tl.math.rsqrt(var)
    tl.store(R + row, r)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        y = x * r * w + 1.0 * w
        tl.store(Y + cols, y, mask=mask)

def calculate_settings(N):
    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4
    return BLOCK_SIZE, num_warps

class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        BLOCK_SIZE, num_warps = calculate_settings(N)
        r = torch.empty((M,), dtype=torch.float32, device="cuda")
        # Apply the Triton kernel
        _rms_layernorm_forward[(M,)](
            x_arg, y, weight, r,
            x_arg.stride(0),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.save_for_backward(x, weight, r)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x, weight, r) = ctx.saved_tensors
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        # Reshape x and dy to 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        dy_arg = dy.reshape(-1, dy.shape[-1])
        M, N = x_arg.shape
        BLOCK_SIZE, num_warps = calculate_settings(N)
        # Call the Triton kernel for backward pass
        _rms_layernorm_backward[(M,)](
            x_arg, dy_arg, dw, dy, weight, r,
            x_arg.stride(0),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return dx, dw

def fast_rms_layernorm(x, weight):
    return Fast_RMS_Layernorm.apply(x, weight)

@
