import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X_ptr, Y_ptr,
    W_ptr, B_ptr,
    X1_ptr, W1_ptr, B1_ptr,
    RES_ptr, Y1_ptr,
    DROPOUT_MASK_ptr,
    SEEDS_ptr,
    MEAN_ptr, INVVAR_ptr,
    M, N,
    P_DROP, ROWSCALE, EPSILON,
    HAS_RESIDUAL, HAS_X1, HAS_W1, HAS_B1, STORE_Y1,
    USE_RMS, STORE_RESIDUAL, STORE_MASK,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    row_offset = pid_m * BLOCK_M
    off_m = row_offset + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    mask_m = off_m < M
    # Pointers
    X_row_ptrs = X_ptr + off_m[:, None] * N + off_n[None, :]
    # Data loads
    x = tl.where(mask_m[:, None], tl.load(X_row_ptrs, mask=None, other=0.0), 0.0)
    # Optional load X1
    if HAS_X1:
        X1_row_ptrs = X1_ptr + off_m[:, None] * N + off_n[None, :]
        x1 = tl.where(mask_m[:, None], tl.load(X1_row_ptrs, mask=None, other=0.0), 0.0)
    else:
        x1 = x  # Unused, for safety.

    # Compute sum for mean or RMS
    psum = tl.sum(x, dims=[1])
    if not USE_RMS:
        mean = psum / N
    else:
        mean = tl.zeros([BLOCK_M], dtype=psum.dtype)

    # Compute sum for variance if not RMS
    if not USE_RMS:
        diff = x - mean[:, None]
        diff_sq = diff * diff
        psum_sq = tl.sum(diff_sq, dims=[1])
        var = psum_sq / N
        invvar = 1.0 / tl.sqrt(var + EPSILON)
    else:
        # RMS uses sqrt( sum(x^2) / N )
        rsum_sq = tl.sum(x * x, dims=[1]) / N
        invvar = 1.0 / tl.sqrt(rsum_sq + EPSILON)

    # Store mean, invvar if needed
    if not USE_RMS:
        tl.store(MEAN_ptr + off_m, mean, mask=mask_m)
    tl.store(INVVAR_ptr + off_m, invvar, mask=mask_m)

    # Load scale/bias
    off_n_scale = off_n
    w = tl.load(W_ptr + off_n_scale, mask=off_n_scale < N, other=0.0)
    b = tl.load(B_ptr + off_n_scale, mask=off_n_scale < N, other=0.0)
    if HAS_W1:
        w1 = tl.load(W1_ptr + off_n_scale, mask=off_n_scale < N, other=0.0)
    else:
        w1 = tl.zeros([BLOCK_N], dtype=w.dtype)
    if HAS_B1:
        b1 = tl.load(B1_ptr + off_n_scale, mask=off_n_scale < N, other=0.0)
    else:
        b1 = tl.zeros([BLOCK_N], dtype=b.dtype)

    # Normalize
    if not USE_RMS:
        x_hat = (x - mean[:, None]) * invvar[:, None]
    else:
        x_hat = x * invvar[:, None]

    # Optional row scale
    if ROWSCALE != 1.0:
        x_hat = ROWSCALE * x_hat

    y = x_hat * w[None, :] + b[None, :]

    # Optional second output
    if STORE_Y1:
        y1 = x_hat * w1[None, :] + b1[None, :]
    else:
        y1 = y  # Unused.

    # Optional dropout
    if P_DROP > 0.0:
        # Generate pseudo-random bitmask
        pid = pid_m
        seed = tl.load(SEEDS_ptr + pid, mask=pid < M, other=0)
        rng = tl.zeros_like(y) + seed
        keep_prob = 1.0 - P_DROP
        rand = tl.rand(rng, off_n_scale)
        dropout_mask = rand < keep_prob
        y = tl.where(dropout_mask, y / keep_prob, 0.0)
        if STORE_Y1:
            y1 = tl.where(dropout_mask, y1 / keep_prob, 0.0)
        # Store dropout mask
        if STORE_MASK:
            mask_ptr = DROPOUT_MASK_ptr + off_m[:, None] * N + off_n[None, :]
            mask_val = tl.where(dropout_mask, 1, 0).to(tl.uint8)
            tl.store(mask_ptr, mask_val, mask=mask_m[:, None] & (off_n < N))

    # Optional residual
    if HAS_RESIDUAL:
        # Only add X to y for example. (Could be extended as needed.)
        y += x
        if STORE_Y1:
            y1 += x
    # Store if user wants residual separately
    if STORE_RESIDUAL:
        res_ptrs = RES_ptr + off_m[:, None] * N + off_n[None, :]
        tl.store(res_ptrs, x, mask=mask_m[:, None] & (off_n < N))

    # Store outputs
    Y_row_ptrs = Y_ptr + off_m[:, None] * N + off_n[None, :]
    tl.store(Y_row_ptrs, y, mask=mask_m[:, None] & (off_n < N))
    if STORE_Y1:
        Y1_row_ptrs = Y1_ptr + off_m[:, None] * N + off_n[None, :]
        tl.store(Y1_row_ptrs, y1, mask=mask_m[:, None] & (off_n < N))


def layer_norm_fwd_1pass(
    X, W, B,
    X1=None, W1=None, B1=None,
    residual_out=None, Y1=None,
    dropout_mask=None, seeds=None,
    p_drop=0.0, row_scale=1.0, epsilon=1e-5,
    has_residual=False, has_x1=False, has_w1=False, has_b1=False, store_y1=False,
    use_rms=False, store_residual=False, store_mask=False
):
    import math
    M, N = X.shape
    BLOCK_M = 1
    BLOCK_N = 128 if N >= 128 else 64
    grid = (math.ceil(M / BLOCK_M),)
    _layer_norm_fwd_1pass_kernel[grid](
        X, 
        Y1 if store_y1 else X,  # Will overwrite in kernel, pass a valid pointer for Y
        W, B,
        X1 if X1 is not None else X,
        W1 if W1 is not None else W,
        B1 if B1 is not None else B,
        residual_out if residual_out is not None else X,
        Y1 if Y1 is not None else X,
        dropout_mask if dropout_mask is not None else X,
        seeds if seeds is not None else X,
        X,               # MEAN placeholder
        X,               # INVVAR placeholder
        M, N,
        p_drop, row_scale, epsilon,
        has_residual, has_x1, has_w1, has_b1, store_y1,
        use_rms, store_residual, store_mask,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
