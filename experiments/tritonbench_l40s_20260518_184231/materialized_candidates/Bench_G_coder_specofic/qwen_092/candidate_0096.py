import triton
import triton.language as tl

BLOCK_SIZE = 32

@triton.jit
def _rms_norm_fwd_fused(x_ptr, weight_ptr, y_ptr, x_shape, weight_shape, n, dtype):
    """
    Triton kernel for RMS normalization.
    """
    # Calculate the index of the current row and column
    row = tl.program_id(0)
    col = tl.program_id(1)
    num_cols = x_shape[1]

    # Load elements from x and weight
    x = tl.load(x_ptr + row * num_cols + col, mask=col < num_cols, other=0.0)
    weight = tl.load(weight_ptr + col, mask=col < num_cols, other=1.0)

    # Compute the variance using blocks of size BLOCK_SIZE
    sum_squares = 0.0
    for i in range(BLOCK_SIZE):
        if col + i < num_cols:
            sum_squares += x * x

    # Reduce sum_squares across the block
    sum_squares = tl.program_sync(sum_squares)

    # Compute the reciprocal standard deviation (rstd)
    variance = sum_squares / num_cols
    rstd = tl.rsqrt(variance + 1e-5)  # Add epsilon to avoid division by zero

    # Normalize and multiply by weight
    normalized_value = x * rstd
    y = normalized_value * weight

    # Store the result in y
    tl.store(y_ptr + row * num_cols + col, y, mask=col < num_cols)
