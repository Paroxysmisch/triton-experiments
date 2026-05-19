import triton
import triton.language as tl
import math


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


def calculate_settings(n_cols, is_hip=False):
    # Determine an appropriate block size and warp configuration
    # BLOCK_SIZE is a power of two based on the number of columns,
    # limited by MAX_FUSED_SIZE or hardware constraints
    MAX_FUSED_SIZE = 4096
    block_size = 1
    while block_size < n_cols and block_size < MAX_FUSED_SIZE:
        block_size <<= 1

    # For different hardware, we can adjust the number of warps
    # to suit the device, here we do a simple heuristic
    num_warps = 4 if not is_hip else 2
    return block_size, num_warps


@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    offset = row_id * n_cols + col_offsets

    mask = col_offsets < n_cols
    a_vals = tl.load(a_ptr + offset, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offset, mask=mask, other=0.0)

    c_vals = b_vals * silu(a_vals)
    tl.store(c_ptr + offset, c_vals, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    a_ptr, b_ptr, dc_ptr,
    da_ptr, db_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    offset = row_id * n_cols + col_offsets

    mask = col_offsets < n_cols
    a_vals = tl.load(a_ptr + offset, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offset, mask=mask, other=0.0)
    dc_vals = tl.load(dc_ptr + offset, mask=mask, other=0.0)

    # Recompute silu(a)
    s = tl.sigmoid(a_vals)
    a_silu = a_vals * s

    # Grad wrt a
    dsilu_da = s + a_vals * s * (1.0 - s)
    da_vals = dc_vals * b_vals * dsilu_da

    # Grad wrt b
    db_vals = dc_vals * a_silu

    tl.store(da_ptr + offset, da_vals, mask=mask)
    tl.store(db_ptr + offset, db_vals, mask=mask)


def swiglu_forward(a, b, is_hip=False):
    """
    a, b: Tensors on GPU
    Returns c = b * silu(a)
    """
    # Shape checks / reshape if needed
    assert a.is_contiguous() and b.is_contiguous()
    n_rows, n_cols = a.shape

    c = a.new_empty(a.shape)
    block_size, num_warps = calculate_settings(n_cols, is_hip)
    grid = (n_rows,)

    _swiglu_forward_kernel[grid](
        a, b, c, n_rows, n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps
    )
    return c


def swiglu_backward(a, b, dc, is_hip=False):
    """
    a, b: original input tensors
    dc: gradient wrt output of swiglu_forward
    Returns da, db: gradient wrt a and b
    """
    assert a.is_contiguous() and b.is_contiguous() and dc.is_contiguous()
    n_rows, n_cols = a.shape

    da = a.new_empty(a.shape)
    db = b.new_empty(b.shape)

    block_size, num_warps = calculate_settings(n_cols, is_hip)
    grid = (n_rows,)

    _swiglu_backward_kernel[grid](
        a, b, dc,
        da, db,
        n_rows, n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps
    )
    return da, db
