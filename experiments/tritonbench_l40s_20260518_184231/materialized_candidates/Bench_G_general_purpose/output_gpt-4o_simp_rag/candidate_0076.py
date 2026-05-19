import torch
import triton
import triton.language as tl

# Triton kernel for LayerNorm
@triton.jit
def _layer_norm_fwd_kernel(
    X,  # pointer to the input tensor
    W,  # pointer to the weights
    Y,  # pointer to the output tensor
    stride_x_N,  # stride for the N dimension of X
    stride_x_hn,  # stride for the H dimension of X
    stride_x_hd,  # stride for the D dimension of X
    stride_y_N,  # stride for the N dimension of Y
    stride_y_hn,  # stride for the H dimension of Y
    stride_y_hd,  # stride for the D dimension of Y
    stride_w_hn,  # stride for the H dimension of W
    stride_w_hd,  # stride for the D dimension of W
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,  # block size for processing
):
    Seq = tl.program_id(0)  # sequence ID
    H = tl.program_id(1)    # height ID

    # Adjust pointers based on the current program IDs
    X += Seq * stride_x_N + H * stride_x_hn
    Y += Seq * stride_y_N + H * stride_y_hn
    W += H * stride_w_hn

    # Compute mean
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Normalize and apply weights
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w

        tl.store(Y + cols, y.to(X.dtype.element_ty), mask=mask)

# Wrapper function for the Triton kernel
def layernorm_forward(
    X,  # input tensor
    W,  # weight tensor
    eps,  # epsilon to avoid division by zero
):
    assert len(X.shape) == 3
    assert len(W.shape) == 2
    assert X.shape[-1] == W.shape[-1]
    assert X.shape[-2] == W.shape[-2]

    y = torch.empty_like(X)

    # Calculate strides for input, output, and weights
    stride_x_N = X.stride(0)
    stride_x_hn = X.stride(1)
    stride_x_hd = X.stride(2)

    stride_y_N = y.stride(0)
    stride_y_hn = y.stride(1)
    stride_y_hd = y.stride(2)

    stride_w_hn = W.stride(0)
    stride_w_hd = W.stride(1)

    N = X.shape[-1]
    BLOCK_SIZE = 128

    # Launch the Triton kernel
    grid = (X.shape[0], X.shape[1])
    _layer_norm_fwd_kernel[grid](
        X,
        W,
        y,
        stride_x_N,
        stride_x_hn,
        stride_x_hd,
        stride_y_N,
        stride_y_hn,
        stride_y_hd,
        stride_w_hn,
        stride_w_hd,
        N,
        eps,
        BLOCK_SIZE,
    )

    return y
