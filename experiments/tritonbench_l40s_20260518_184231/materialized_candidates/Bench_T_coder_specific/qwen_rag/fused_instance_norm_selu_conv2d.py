@triton.jit
def conv2d_kernel(
    input,
    weight,
    output,
    input_row_stride,
    input_col_stride,
    output_row_stride,
    output_col_stride,
    input_rows,
    input_cols,
    kernel_rows,
    kernel_cols,
    groups,
    stride_rows,
    stride_cols,
    padding_rows,
    padding_cols,
    dilation_rows,
    dilation_cols,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_m = pid % (input_rows // stride_rows)
    block_n = pid // (input_rows // stride_rows)
    
    row_start = block_m * stride_rows - padding_rows
    col_start = block_n * stride_cols - padding_cols
    
    row_end = min(row_start + kernel_rows, input_rows)
    col_end = min(col_start + kernel_cols, input_cols)
    
    row_range = tl.arange(0, kernel_rows)
    col_range = tl.arange(0, kernel_cols)
    
    m = row_start + row_range[:, None] * dilation_rows
    n = col_start + col_range[None, :] * dilation_cols
    
    mask = ((m >= 0) & (m < input_rows)) & ((n >= 0) & (n < input_cols))
    
    x = tl.load(input + m * input_row_stride + n * input_col_stride, mask=mask, partial=True)
    w = tl.load(weight + pid * kernel_rows * kernel_cols, mask=mask, partial=True)
    
    acc = tl.zeros((N_ROWS,), dtype=input.dtype)
    for k in range(kernel_rows * kernel_cols):
        acc += x[:, k] * w[k]
    
    o_ptr = output + block_m * output_row_stride + block_n * output_col_stride
    tl.store(o_ptr, acc, mask=mask)
