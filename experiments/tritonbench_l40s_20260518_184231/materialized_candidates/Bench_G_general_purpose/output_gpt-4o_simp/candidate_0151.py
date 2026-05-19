import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X, Y, W, B, 
    RESIDUAL=None, X1=None, W1=None, B1=None,
    y_ptr=None, y1_ptr=None, mean_ptr=None, rstd_ptr=None,
    residual_out_ptr=None, seeds_ptr=None, dropout_mask_ptr=None, dropout_mask1_ptr=None,
    NORM_SIZE: tl.constexpr, IS_RMS_NORM: tl.constexpr
):
    # Compute the mean and variance
    row_id = tl.program_id(0)
    offset = row_id * NORM_SIZE
    x = tl.load(X + offset, mask=True)

    # Compute mean
    mean = tl.sum(x, axis=0) / NORM_SIZE
    if not IS_RMS_NORM:
        # Compute variance
        var = tl.sum((x - mean) ** 2, axis=0) / NORM_SIZE
        rstd = 1.0 / tl.sqrt(var + 1e-5)
    else:
        rstd = 1.0

    # Normalize
    x_norm = (x - mean) * rstd

    # Apply weights and biases
    w = tl.load(W, mask=True)
    b = tl.load(B, mask=True)
    y = x_norm * w + b

    # Optional element-wise operations with additional tensors
    if X1 is not None and W1 is not None and B1 is not None:
        x1 = tl.load(X1 + offset, mask=True)
        w1 = tl.load(W1, mask=True)
        b1 = tl.load(B1, mask=True)
        y1 = x1 * w1 + b1
        if y1_ptr is not None:
            tl.store(y1_ptr + offset, y1, mask=True)

    # Optional dropout
    if dropout_mask_ptr is not None:
        dropout_mask = tl.load(dropout_mask_ptr + offset, mask=True)
        y = y * dropout_mask

    # Store the results
    tl.store(y_ptr + offset, y, mask=True)
    if mean_ptr is not None:
        tl.store(mean_ptr + row_id, mean, mask=True)
    if rstd_ptr is not None:
        tl.store(rstd_ptr + row_id, rstd, mask=True)
    if residual_out_ptr is not None and RESIDUAL is not None:
        residual = tl.load(RESIDUAL + offset, mask=True)
        residual_out = y + residual
        tl.store(residual_out_ptr + offset, residual_out, mask=True)

def layer_norm_fwd_1pass(X, W, B, RESIDUAL=None, X1=None, W1=None, B1=None, IS_RMS_NORM=False):
    # Allocate output tensors
    Y = triton.zeros_like(X)
    Y1 = triton.zeros_like(X1) if X1 is not None else None
    mean = triton.empty((X.shape[0],), dtype=triton.float32)
    rstd = triton.empty((X.shape[0],), dtype=triton.float32)
    residual_out = triton.zeros_like(X) if RESIDUAL is not None else None

    # Launch the kernel
    grid = (X.shape[0],)
    _layer_norm_fwd_1pass_kernel[grid](
        X, Y, W, B, 
        RESIDUAL, X1, W1, B1,
        Y, Y1, mean, rstd,
        residual_out, None, None, None,
        NORM_SIZE=X.shape[1], IS_RMS_NORM=IS_RMS_NORM
    )

    return Y, Y1, mean, rstd, residual_out

# Example usage
# X, W, B, etc. are Triton tensors
# Y, Y1, mean, rstd, residual_out = layer_norm_fwd_1pass(X, W, B)
