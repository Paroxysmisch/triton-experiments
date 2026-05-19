import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    M, N,
    RESIDUAL_ptr=None, X1_ptr=None, W1_ptr=None, B1_ptr=None,
    ROWSCALE=None, SEEDS=None, DROPOUT_MASK_ptr=None,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    RMS_NORM: tl.constexpr = False
):
    pid = tl.program_id(0)
    num_pid_m = M // BLOCK_SIZE_M
    num_pid_n = N // BLOCK_SIZE_N

    # Compute indices for this program instance
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Block-level memory pointers
    X_block_ptr = X_ptr + pid_m * BLOCK_SIZE_M * N + pid_n * BLOCK_SIZE_N
    Y_block_ptr = Y_ptr + pid_m * BLOCK_SIZE_M * N + pid_n * BLOCK_SIZE_N
    W_block_ptr = W_ptr + pid_n * BLOCK_SIZE_N
    B_block_ptr = B_ptr + pid_n * BLOCK_SIZE_N

    # Load data
    X = tl.load(X_block_ptr, mask=(pid_m < M) & (pid_n < N))
    W = tl.load(W_block_ptr)
    B = tl.load(B_block_ptr)

    # Compute mean and variance (if not RMS)
    if not RMS_NORM:
        mean = tl.sum(X, axis=1) / N
        var = tl.sum((X - mean[:, None])**2, axis=1) / N
        inv_std = 1 / tl.sqrt(var + 1e-5)
    else:
        inv_std = 1 / tl.sqrt(tl.sum(X**2, axis=1) / N + 1e-5)

    # Normalize
    X_norm = (X - mean[:, None]) * inv_std[:, None] if not RMS_NORM else X * inv_std[:, None]

    # Apply weights and biases
    Y = X_norm * W + B

    # Handle optional RESIDUAL, X1, W1, B1
    if RESIDUAL_ptr is not None:
        RESIDUAL = tl.load(RESIDUAL_ptr + pid_m * BLOCK_SIZE_M * N + pid_n * BLOCK_SIZE_N)
        Y += RESIDUAL

    if X1_ptr is not None and W1_ptr is not None and B1_ptr is not None:
        X1 = tl.load(X1_ptr + pid_m * BLOCK_SIZE_M * N + pid_n * BLOCK_SIZE_N)
        W1 = tl.load(W1_ptr + pid_n * BLOCK_SIZE_N)
        B1 = tl.load(B1_ptr + pid_n * BLOCK_SIZE_N)
        Y1 = X1 * W1 + B1
        Y += Y1

    # Apply dropout if SEEDS and DROPOUT_MASK are provided
    if SEEDS is not None and DROPOUT_MASK_ptr is not None:
        seed = SEEDS[pid]
        dropout_mask = tl.random(seed, Y.shape) < 0.5  # Assuming 50% dropout
        Y *= dropout_mask
        tl.store(DROPOUT_MASK_ptr + pid_m * BLOCK_SIZE_M * N + pid_n * BLOCK_SIZE_N, dropout_mask)

    # Store results
    tl.store(Y_block_ptr, Y)

def layer_norm_fwd_1pass(
    X, W, B, M, N,
    RESIDUAL=None, X1=None, W1=None, B1=None,
    ROWSCALE=None, SEEDS=None, DROPOUT_MASK=None,
    block_size_m=128, block_size_n=128, rms_norm=False
):
    grid = lambda META: (triton.cdiv(M, block_size_m) * triton.cdiv(N, block_size_n),)
    Y = torch.empty_like(X)
    _layer_norm_fwd_1pass_kernel[grid](
        X, Y, W, B, M, N,
        RESIDUAL, X1, W1, B1,
        ROWSCALE, SEEDS, DROPOUT_MASK,
        BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=block_size_n,
        RMS_NORM=rms_norm
    )
    return Y
