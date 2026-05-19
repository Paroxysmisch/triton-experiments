import math
import triton
import triton.language as tl


# ---------------------------------------------------------------------------------
# FORWARD KERNEL
# ---------------------------------------------------------------------------------
@triton.jit
def _geglu_tanh_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_a, stride_b, stride_c,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for the row index
    row_id = tl.program_id(0)
    # Compute the start index for this row in each tensor
    offset_a = row_id * stride_a
    offset_b = row_id * stride_b
    offset_c = row_id * stride_c

    # Create a range of column indices for this program
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Load a and b (if in range)
    a_vals = tl.load(a_ptr + offset_a + cols, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offset_b + cols, mask=mask, other=0.0)

    # Constants for approximate GELU
    sqrt_2_over_pi = math.sqrt(2 / math.pi)
    alpha = 0.044715

    # gelu_approx(b) = 0.5 * b * (1 + tanh(sqrt(2/pi)*(b + 0.044715*b^3)))
    b_cubed = b_vals * b_vals * b_vals
    inner = sqrt_2_over_pi * (b_vals + alpha * b_cubed)
    t = tl.math.tanh(inner)
    gelu_b = 0.5 * b_vals * (1.0 + t)

    # GEGLU: c = a * gelu(b)
    c_vals = a_vals * gelu_b

    # Store the result
    tl.store(c_ptr + offset_c + cols, c_vals, mask=mask)


# ---------------------------------------------------------------------------------
# BACKWARD KERNEL
# ---------------------------------------------------------------------------------
@triton.jit
def _geglu_tanh_backward_kernel(
    a_ptr, b_ptr, dc_ptr,
    da_ptr, db_ptr,
    stride_a, stride_b, stride_c,
    stride_da, stride_db,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for the row index
    row_id = tl.program_id(0)
    # Compute the start index for this row in each tensor
    offset_a = row_id * stride_a
    offset_b = row_id * stride_b
    offset_dc = row_id * stride_c
    offset_da = row_id * stride_da
    offset_db = row_id * stride_db

    # Create a range of column indices for this program
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Load input, b, and upstream grads
    a_vals = tl.load(a_ptr + offset_a + cols, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offset_b + cols, mask=mask, other=0.0)
    dc_vals = tl.load(dc_ptr + offset_dc + cols, mask=mask, other=0.0)

    # Constants for approximate GELU
    sqrt_2_over_pi = math.sqrt(2 / math.pi)
    alpha = 0.044715

    # Recompute tanh part
    b_cubed = b_vals * b_vals * b_vals
    inner = sqrt_2_over_pi * (b_vals + alpha * b_cubed)
    t = tl.math.tanh(inner)

    # gelu_approx(b) = 0.5 * b * (1 + t)
    gelu_b = 0.5 * b_vals * (1.0 + t)

    # d/d(b) gelu_approx(b)
    # = 0.5 * (1 + t) + 0.5 * b * (1 - t^2) * sqrt_2_over_pi * (1 + 3*alpha*b^2)
    dt = 0.5 * (1.0 + t) + 0.5 * b_vals * (1.0 - t * t) * sqrt_2_over_pi * (1.0 + 3.0 * alpha * (b_vals * b_vals))

    # Grad w.r.t. a: da = dc * gelu(b)
    da_vals = dc_vals * gelu_b
    # Grad w.r.t. b: db = dc * a * d/d(b) gelu_approx(b)
    db_vals = dc_vals * a_vals * dt

    # Store gradients
    tl.store(da_ptr + offset_da + cols, da_vals, mask=mask)
    tl.store(db_ptr + offset_db + cols, db_vals, mask=mask)


# ---------------------------------------------------------------------------------
# FORWARD WRAPPER
# ---------------------------------------------------------------------------------
def geglu_forward(a, b):
    """
    a, b: 2D tensors (n_rows, n_cols)
    Returns: c = a * approx_gelu(b), with shape (n_rows, n_cols)
    """
    assert a.shape == b.shape, "Input shapes must match."
    n_rows, n_cols = a.shape

    # Allocate output
    c = a.new_empty(a.shape)

    # Kernel execution parameters
    BLOCK_SIZE = 128
    num_warps = 4

    # Launch
    grid = (n_rows,)
    _geglu_tanh_forward_kernel[grid](
        a, b, c,
        a.stride(0), b.stride(0), c.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return c


# ---------------------------------------------------------------------------------
# BACKWARD WRAPPER
# ---------------------------------------------------------------------------------
def geglu_backward(a, b, dc):
    """
    a, b: 2D tensors (n_rows, n_cols), dc: 2D tensor for upstream gradient
    Returns: gradients w.r.t. a and b
    """
    assert a.shape == b.shape == dc.shape, "All shapes must match."
    n_rows, n_cols = a.shape

    da = a.new_empty(a.shape)
    db = b.new_empty(b.shape)

    # Kernel execution parameters
    BLOCK_SIZE = 128
    num_warps = 4

    # Launch
    grid = (n_rows,)
    _geglu_tanh_backward_kernel[grid](
        a, b, dc,
        da, db,
        a.stride(0), b.stride(0), dc.stride(0),
        da.stride(0), db.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return da, db
