import torch
import triton
import triton.language as tl


@triton.jit
def q_kernel_per_block_int8(
    q_ptr, q_int8_ptr, q_scale_ptr,
    rows, cols,
    stride_q_col, stride_q_int8_col, stride_scale,
    BLOCK_SIZE_ROWS: tl.constexpr, BLOCK_SIZE_COLS: tl.constexpr
):
    """
    Triton kernel to quantize a block of the query matrix into int8 with scaling.
    rows, cols: dimensions of the 2D (reshaped) query tensor
    stride_q_col: stride for query in the column dimension
    stride_q_int8_col: stride for int8 query in the column dimension
    stride_scale: stride for the scale tensor in the row dimension
    BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS: block sizes in rows & columns
    """

    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)

    row_start = pid_row * BLOCK_SIZE_ROWS
    col_start = pid_col * BLOCK_SIZE_COLS

    # Create index maps for rows/columns within a block
    block_row_indices = row_start + tl.arange(0, BLOCK_SIZE_ROWS)
    block_col_indices = col_start + tl.arange(0, BLOCK_SIZE_COLS)

    # Masks to guard memory loads/stores
    row_mask = block_row_indices < rows
    col_mask = block_col_indices < cols

    # Initialize max_abs accumulator
    max_abs_block = tl.zeros((BLOCK_SIZE_ROWS,), dtype=tl.float32)

    # Compute max absolute value across the block
    for c_ofs in range(0, BLOCK_SIZE_COLS):
        c_idx = col_start + c_ofs
        col_valid = c_idx < cols
        q_ptrs = q_ptr + (block_row_indices * stride_q_col) + c_idx
        vals = tl.where(
            row_mask & col_valid,
            tl.load(q_ptrs, mask=row_mask & col_valid, other=0.0),
            0.0
        )
        max_abs_block = tl.maximum(max_abs_block, tl.abs(vals))

    # Reduce maximum across all rows in this block to get a single scale
    block_max_abs = 0.0
    for i in range(BLOCK_SIZE_ROWS):
        if i < BLOCK_SIZE_ROWS:
            block_max_abs = tl.max(block_max_abs, max_abs_block[i])
    block_max_abs = tl.maximum(block_max_abs, 1e-8)  # Avoid div-by-zero
    scale = 127.0 / block_max_abs

    # Store the scale (one scale per block-row)
    # Each row-block gets one scale entry
    scale_ptr = q_scale_ptr + row_start
    if row_mask[0]:
        tl.store(scale_ptr, scale, mask=row_mask[0])

    # Quantize the block
    for c_ofs in range(0, BLOCK_SIZE_COLS):
        c_idx = col_start + c_ofs
        col_valid = c_idx < cols
        q_ptrs = q_ptr + (block_row_indices * stride_q_col) + c_idx
        vals = tl.where(
            row_mask & col_valid,
            tl.load(q_ptrs, mask=row_mask & col_valid, other=0.0),
            0.0
        )
        # Multiply by scale, round to nearest int
        q8_vals = tl.round(vals * scale)
        # Clamp to [-127, 127]
        q8_vals = tl.maximum(q8_vals, -127.0)
        q8_vals = tl.minimum(q8_vals, 127.0)
        # Cast to int8
        q8_vals_int8 = q8_vals.to(tl.int8)

        q_int8_ptrs = q_int8_ptr + (block_row_indices * stride_q_int8_col) + c_idx
        tl.store(q_int8_ptrs, q8_vals_int8, mask=row_mask & col_valid)


@triton.jit
def k_kernel_per_block_int8(
    k_ptr, k_int8_ptr, k_scale_ptr,
    rows, cols,
    stride_k_col, stride_k_int8_col, stride_scale,
    BLOCK_SIZE_ROWS: tl.constexpr, BLOCK_SIZE_COLS: tl.constexpr
):
    """
    Triton kernel to quantize a block of the key matrix into int8 with scaling.
    rows, cols: dimensions of the 2D (reshaped) key tensor
    stride_k_col: stride for key in the column dimension
    stride_k_int8_col: stride for int8 key in the column dimension
    stride_scale: stride for the scale tensor in the row dimension
    BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS: block sizes in rows & columns
    """

    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)

    row_start = pid_row * BLOCK_SIZE_ROWS
    col_start = pid_col * BLOCK_SIZE_COLS

    block_row_indices = row_start + tl.arange(0, BLOCK_SIZE_ROWS)
    block_col_indices = col_start + tl.arange(0, BLOCK_SIZE_COLS)

    row_mask = block_row_indices < rows
    col_mask = block_col_indices < cols

    max_abs_block = tl.zeros((BLOCK_SIZE_ROWS,), dtype=tl.float32)

    # Compute max absolute value across the block
    for c_ofs in range(0, BLOCK_SIZE_COLS):
        c_idx = col_start + c_ofs
        col_valid = c_idx < cols
        k_ptrs = k_ptr + (block_row_indices * stride_k_col) + c_idx
        vals = tl.where(
            row_mask & col_valid,
            tl.load(k_ptrs, mask=row_mask & col_valid, other=0.0),
            0.0
        )
        max_abs_block = tl.maximum(max_abs_block, tl.abs(vals))

    block_max_abs = 0.0
    for i in range(BLOCK_SIZE_ROWS):
        block_max_abs = tl.max(block_max_abs, max_abs_block[i])
    block_max_abs = tl.maximum(block_max_abs,
