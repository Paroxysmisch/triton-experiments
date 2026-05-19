import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_xD1, stride_xD2, stride_xN,
    stride_wD2, stride_wN,
    stride_yD1, stride_yD2, stride_yN,
    D1, D2, N,
    BLOCK_SIZE,
    EPS,
    **meta
):
    # Program IDs for the first two dimensions (D1 and D2)
    d1 = tl.program_id(0)
    d2 = tl.program_id(1)

    # Pointers for the corresponding 2D slice
    X_row_ptr = X_ptr + d1 * stride_xD1 + d2 * stride_xD2
    W_row_ptr = W_ptr + d2 * stride_wD2
    Y_row_ptr = Y_ptr + d1 * stride_yD1 + d2 * stride_yD2

    # Accumulate sum and sum of squares (in float32 for numerical stability)
    sum_x = tl.zeros([], dtype=tl.float32)
    sum_x2 = tl.zeros([], dtype=tl.float32)

    # First pass: compute mean and variance
    # Loop over N in chunks of BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    for i in range(0, N, BLOCK_SIZE):
        idx = i + offs
        mask = idx < N
        x = tl.load(X_row_ptr + idx * stride_xN, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, where=mask)
        sum_x2 += tl.sum(x * x, where=mask)

    # Final mean and variance
    # Since we've accumulated over all elements along the last dimension
    denom = tl.float32(N)
    mean = sum_x / denom
    var = (sum_x2 / denom) - mean * mean
    var = tl.maximum(var, 0.0)  # numerical clamp
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Second pass: write normalized output * W
    for i in range(0, N, BLOCK_SIZE):
        idx = i + offs
        mask = idx < N
        x = tl.load(X_row_ptr + idx * stride_xN, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_row_ptr + idx * stride_wN, mask=mask, other=0.0).to(tl.float32)

        # Normalize and apply weight
        y = (x - mean) * inv_std * w
        # Store result
        tl.store(Y_row_ptr + idx * stride_yN, y, mask=mask)


def layernorm_forward(X: torch.Tensor, W: torch.Tensor, block_size=128, eps=1e-5):
    """
    Forward pass of layer normalization:
    X: (D1, D2, N)
    W: (D2, N)
    Returns Y: (D1, D2, N)
    """
    D1, D2, N = X.shape
    assert W.shape == (D2, N), "Weight shape must match (D2, N)"

    Y = torch.empty_like(X)

    # Strides
    stride_xD1 = X.stride(0)
    stride_xD2 = X.stride(1)
    stride_xN  = X.stride(2)
    stride_wD2 = W.stride(0)
    stride_wN  = W.stride(1)
    stride_yD1 = Y.stride(0)
    stride_yD2 = Y.stride(1)
    stride_yN  = Y.stride(2)

    # Launch grid = (D1, D2)
    grid = (D1, D2)

    _layer_norm_fwd_kernel[grid](
        X, W, Y,
        stride_xD1, stride_xD2, stride_xN,
        stride_wD2, stride_wN,
        stride_yD1, stride_yD2, stride_yN,
        D1, D2, N,
        block_size,
        eps,
        num_warps=4
    )

    return Y
