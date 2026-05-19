import torch
import triton
import triton.language as tl

@triton.jit
def calculate_settings(n):
    # Calculate the maximum number of columns that can be processed in a single block
    max_fused_size = triton.next_power_of_2(n)
    MAX_FUSED_SIZE = 64 * 1024
    if max_fused_size > MAX_FUSED_SIZE:
        max_fused_size = MAX_FUSED_SIZE

    # Determine the number of warps and the size of each block
    num_warps = 4
    if max_fused_size >= 2 * 1024:
        num_warps = 8
    if max_fused_size >= 4 * 1024:
        num_warps = 16

    BLOCK_SIZE = max_fused_size
    return BLOCK_SIZE, num_warps

@triton.jit
def layernorm_forward(
    X,
    Y,
    r,
    mean,
    rstd,
    W,
    b,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the row index and the column offsets for the block
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load the row and its normalization factors
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0)
    _mean = tl.sum(x * mask, axis=0) / N
    x_zm = tl.where(mask, x - _mean, 0.0)
    var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(r + row, rstd)

    # Normalize and apply linear transformation
    x_hat = x_zm * rstd
    y = x_hat * tl.load(W + cols, mask=mask) + tl.load(b + cols, mask=mask)
    tl.store(Y + row * stride + cols, y, mask=mask)

@triton.jit
def layernorm_backward(
    dY,
    X,
    r,
    dX,
    mu,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the row index and the column offsets for the block
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load the inputs and normalization factors
    _dY = tl.load(dY + row * N + cols, mask=mask, other=0)
    x = tl.load(X + row * N + cols, mask=mask, other=0)
    _r = tl.load(r + row)
    _mu = tl.load(mu + row)

    # Compute intermediate quantities needed for the gradient
    x_zm = tl.where(mask, x - _mu, 0.0)
    rstd = 1 / _r
    df = _dY * rstd
    dx = df * tl.where(mask, 1 - rstd * x_zm * rstd, 0.0)
    m1 = tl.sum(dx * x_zm, axis=0) / N
    m2 = tl.sum(dx, axis=0) / N
    tl.store(dX + row * N + cols, dx, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def fast_layernorm(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    x_ptr_grad,
    mean_ptr,
    inv_std_ptr,
    norm_shape,
    eps,
    stride,
    N: tl.constexpr,
    M: tl.constexpr,
):
    # Define constants and calculate settings for the layer norm
    BLOCK_SIZE, num_warps = calculate_settings(N)
    pre_computed = tl.zeros((M, 2), dtype=tl.float32)
    pid = tl.program_id(0)

    # Iterate through the input tensor to process blocks
    for i in range(0, M, BLOCK_SIZE):
        row = pid * BLOCK_SIZE + i + tl.arange(0, BLOCK_SIZE)
        mask = row < M

        x = tl.load(x_ptr + row * stride, mask=mask, other=0.0).to(tl.float32)
        layernorm_forward(
            x,
            y_ptr,
            pre_computed,
            mean_ptr,
            inv_std_ptr,
            w_ptr,
            b_ptr,
            stride,
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        layernorm_backward(
            y_ptr,
            x,
            pre_computed,
            x_ptr_grad,
            mean_ptr,
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        # Prepare output tensors and call the Triton kernel
        y = torch.empty_like(x)
        x_grad = torch.empty_like(x)
        M, N = x.shape
        mean = torch.empty((M,), dtype=torch.float32, device="cuda")
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")

        pre_computed = torch.empty((M, 2), dtype=torch.float32, device="cuda")
        assert x.is_contiguous()

        norm_shape = x.shape[-1]
        M = x.numel() // norm_shape

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]),)
        fast_layernorm[grid](
            x,
            y,
            weight,
            bias,
            x_grad,
            mean,
            rstd,
            norm_shape,
            eps,
            stride=x.stride(0),
            N=N,
            M=M,
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE, ctx.num_warps = calculate_settings(N)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        # Compute the gradient of the input using the Triton kernel
        x, w, b, m, v = ctx.saved_tensors
        N = w.shape[0]
        M = x.shape[0]

        assert dy.is_contiguous()
        dx = torch.empty_like(x)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]),)
        layernorm_backward[grid](
            dy,
            x,
            v,
            dx,
            m,
            N,
            eps=ctx.eps,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )
        return dx, None, None, None

def fast_layer_norm(layernorm):
    # Create a custom layer norm function using the autograd function
    def fast_layer_norm(x):
        assert x.is_contiguous()
        M, N = x.shape
        out_shape = x.shape
        x = x.view(-1, N)
        M, N = x.shape
        weight = layernorm.weight
        bias = layernorm.bias
        out = Fast_Layernorm.apply(x, weight, bias, layernorm.eps)
        out = out.view(out_shape)
        return out

    return fast_layer_norm
