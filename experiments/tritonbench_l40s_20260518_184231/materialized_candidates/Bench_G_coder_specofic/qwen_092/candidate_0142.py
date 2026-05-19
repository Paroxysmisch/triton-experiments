triton
import triton
import triton.language as tl

# Constants
BLOCK_SIZE = 128
NUM_WARPS = 4

@triton.jit
def _geglu_tanh_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    row_end = min(row_start + BLOCK_SIZE, n_rows)
    row_range = tl.arange(row_start, row_end)
    col_range = tl.arange(n_cols)

    a = tl.load(a_ptr + row_range * n_cols[:, None] + col_range)
    b = tl.load(b_ptr + row_range * n_cols[:, None] + col_range)

    # GEGLU activation using tanh approximation
    tanh_approx = 0.5 * a * (1 + tl.tanh(tl.sqrt(2.0 / tl.f32(3.14159)) * (a + 0.044715 * a * a * a)))
    c = a * tanh_approx + b * (1 - tanh_approx)

    tl.store(c_ptr + row_range * n_cols[:, None] + col_range, c)

@triton.jit
def _geglu_tanh_backward_kernel(
    dc_ptr, a_ptr, b_ptr, da_ptr, db_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    row_end = min(row_start + BLOCK_SIZE, n_rows)
    row_range = tl.arange(row_start, row_end)
    col_range = tl.arange(n_cols)

    dc = tl.load(dc_ptr + row_range * n_cols[:, None] + col_range)
    a = tl.load(a_ptr + row_range * n_cols[:, None] + col_range)
    b = tl.load(b_ptr + row_range * n_cols[:, None] + col_range)

    # Derivative of GEGLU activation using tanh approximation
    tanh_approx = 0.5 * a * (1 + tl.tanh(tl.sqrt(2.0 / tl.f32(3.14159)) * (a + 0.044715 * a * a * a)))
    tanh_prime = 0.5 * (1 + tl.tanh(tl.sqrt(2.0 / tl.f32(3.14159)) * (a + 0.044715 * a * a * a))) * (1 - tl.tanh(tl.sqrt(2.0 / tl.f32(3.14159)) * (a + 0.044715 * a * a * a))) * (1 + 3 * 0.044715 * a * a)
    tanh_approx_prime = 0.5 * (1 + tl.tanh(tl.sqrt(2.0 / tl.f32(3.14159)) * (a + 0.044715 * a * a * a))) * (1 - tl.tanh(tl.sqrt(2.0 / tl.f32(3.14159)) * (a + 0.044715 * a * a * a))) * (1 + 3 * 0.044715 * a * a)
    da = dc * (tanh_approx_prime * a + (1 - tanh_approx) * (1 + 0.044715 * a * a * a))
    db = dc * (1 - tanh_approx)

    tl.store(da_ptr + row_range * n_cols[:, None] + col_range, da)
    tl.store(db_ptr + row_range * n_cols[:, None] + col_range, db)

@triton.jit
def gegluf_forward(a, b, n_rows, n_cols):
    # Reshape a and b to 2D format
    a = a.reshape(-1, n_cols)
    b = b.reshape(-1, n_cols)

    # Initialize output tensor c
    c = tl.zeros((n_rows, n_cols), dtype=tl.float32)

    # Calculate BLOCK_SIZE and num_warps
    num_warps = NUM_WARPS
    grid_size = (n_rows + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch forward kernel
    _geglu_tanh_forward_kernel[grid_size, BLOCK_SIZE, num_warps](a, b, c, n_rows, n_cols)

    return c

@triton.jit
def gegluf_backward(dc, a, b, n_rows, n_cols):
    # Reshape dc, a, and b to 2D format
    dc = dc.reshape(-1, n_cols)
    a = a.reshape(-1, n_cols)
    b = b.reshape(-1, n_cols)

    # Initialize gradients tensors da and db
    da = tl.zeros((n_rows, n_cols), dtype=tl.float32)
    db = tl.zeros((n_rows, n_cols), dtype=tl.float32)

    # Calculate BLOCK_SIZE and num_warps
    num_warps = NUM_WARPS
    grid_size = (n_rows + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch backward kernel
    _geglu_tanh_backward_kernel[grid_size, BLOCK_SIZE, num_warps](dc, a, b, da, db, n_rows, n_cols)

    return da, db
