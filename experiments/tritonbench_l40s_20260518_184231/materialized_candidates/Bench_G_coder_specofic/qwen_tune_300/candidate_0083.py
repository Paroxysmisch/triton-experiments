import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_x_col,  # how much to increase the pointer when moving by 1 col
    stride_y_row,
    stride_y_col,
    stride_w_row,
    stride_w_col,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col in range(0, N, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols * stride_x_col, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col in range(0, N, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols * stride_x_col, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols * stride_w_col, mask=mask).to(tl.float32)
        x = tl.load(X + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols * stride_y_col, y, mask=mask)

def layernorm_forward(x, weight, eps):
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.view(-1, x.shape[-1])
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    grid = (M, 1)
    # enqueue kernel
    _layer_norm_fwd_kernel[grid](
        x_arg,
        y,
        weight,
        x_arg.stride(0),
        x_arg.stride(1),
        y.stride(0),
        y.stride(1),
        weight.stride(0),
        weight.stride(1),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y
