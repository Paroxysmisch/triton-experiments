import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X, Y, W, B, X1, W1, B1, ROWSCALE, SEEDS, DROPOUT_MASK, residual_out, M, N, HEADS, RESIDUAL, RMS_NORM,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Define constants
    BLOCK_SIZE_K = 16  # Assuming K is always 16 for simplicity

    # Get the indices for the current block
    pid = tl.program_id(axis=0)
    pid_m = pid // (BLOCK_SIZE_N * HEADS)
    pid_n = (pid % (BLOCK_SIZE_N * HEADS)) // BLOCK_SIZE_N
    pid_k = pid % BLOCK_SIZE_N

    # Calculate the global indices
    row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = pid_n * BLOCK_SIZE_N + pid_k

    # Initialize accumulators
    mean = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    var = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Load data
    x = tl.load(X + row[:, None] * N + col[None, :])
    w = tl.load(W + pid_k)
    b = tl.load(B + pid_k)

    # Compute mean and variance
    for k in range(BLOCK_SIZE_K):
        x_k = x[:, k]
        mean += x_k
        var += x_k * x_k

    # Normalize
    mean = mean / BLOCK_SIZE_K
    var = var / BLOCK_SIZE_K - mean * mean

    # Apply weights and biases
    y = x * w + b

    # Handle residual connections
    if RESIDUAL:
        x1 = tl.load(X1 + row[:, None] * N + col[None, :])
        w1 = tl.load(W1 + pid_k)
        b1 = tl.load(B1 + pid_k)
        y += x1 * w1 + b1

    # Handle RMS normalization
    if RMS_NORM:
        var = tl.sqrt(var)

    # Apply dropout
    if SEEDS is not None:
        seed = tl.load(SEEDS + row[:, None] * N + col[None, :])
        dropout_mask = tl.load(DROPOUT_MASK + row[:, None] * N + col[None, :])
        y *= dropout_mask

    # Store results
    tl.store(Y + row[:, None] * N + col[None, :], y)

    # Store mean and variance if not RMS normalization
    if not RMS_NORM:
        tl.store(Y + row[:, None] * N + (col[None, :] + N), mean)
        tl.store(Y + row[:, None] * N + (col[None, :] + 2 * N), var)

    # Store residuals if specified
    if residual_out is not None:
        tl.store(residual_out + row[:, None] * N + col[None, :], y - x)

def layer_norm_fwd_1pass(X, Y, W, B, X1=None, W1=None, B1=None, ROWSCALE=None, SEEDS=None, DROPOUT_MASK=None, residual_out=None, M=None, N=None, HEADS=None, RESIDUAL=False, RMS_NORM=False, BLOCK_SIZE_M=128, BLOCK_SIZE_N=64):
    # Check input shapes
    assert X.shape == (M, N)
    assert Y.shape == (M, N)
    assert W.shape == (N,)
    assert B.shape == (N,)
    if X1 is not None:
        assert X1.shape == (M, N)
    if W1 is not None:
        assert W1.shape == (N,)
    if B1 is not None:
        assert B1.shape == (N,)
    if ROWSCALE is not None:
        assert ROWSCALE.shape == (M, N)
    if SEEDS is not None:
        assert SEEDS.shape == (M, N)
    if DROPOUT_MASK is not None:
        assert DROPOUT_MASK.shape == (M, N)
    if residual_out is not None:
        assert residual_out.shape == (M, N)

    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N) * HEADS, 1)
    _layer_norm_fwd_1pass_kernel[grid](X, Y, W, B, X1, W1, B1, ROWSCALE, SEEDS, DROPOUT_MASK, residual_out, M, N, HEADS, RESIDUAL, RMS_NORM, BLOCK_SIZE_M, BLOCK_SIZE_N)
