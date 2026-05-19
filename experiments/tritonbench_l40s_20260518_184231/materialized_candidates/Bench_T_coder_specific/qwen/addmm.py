import triton
import triton.language as tl

@triton.jit
def addmm_kernel(
    input_ptr, mat1_ptr, mat2_ptr, output_ptr,
    n_rows: tl.int32, n_cols: tl.int32, k: tl.int32,
    alpha: tl.float32, beta: tl.float32,
    BLOCK_SIZE: tl.constexpr(32),
):
    pid = tl.program_id(axis=0)
    block_row_start = pid * BLOCK_SIZE
    block_col_start = tl.program_id(axis=1) * BLOCK_SIZE
    
    row = block_row_start + tl.arange(0, BLOCK_SIZE)
    col = block_col_start + tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Perform matrix multiplication and accumulation
    for m in range(0, k, BLOCK_SIZE):
        mat1_block = tl.load(mat1_ptr + (row[:, None] * k + m)[None, :] * n_cols + col[None, :])
        mat2_block = tl.load(mat2_ptr + (m[:, None] * n_cols + col)[None, :] * k + row[None, :])
        acc += alpha * mat1_block @ mat2_block
    
    # Accumulate into the output buffer
    for m in range(0, k, BLOCK_SIZE):
        acc_block = acc[:, :]
        output_ptr += (block_row_start + row[:, None]) * n_cols + (block_col_start + col[None, :])
        tl.store(output_ptr, beta * tl.load(output_ptr) + alpha * acc_block)

# Wrapper Function
def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    assert input.ndim == 2 and mat1.ndim == 2 and mat2.ndim == 2, "Input tensors must be 2D"
    assert mat1.shape[1] == mat2.shape[0], "Matrix dimensions must match for multiplication"

    n_rows, n_cols = input.shape
    k = mat1.shape[1]

    if out is None:
        out = tl.zeros_like(input)

    assert out.shape == (n_rows, n_cols), "Output shape must match input shape"
    assert out.dtype == input.dtype, "Output dtype must match input dtype"

    grid = (
        triton.cdiv(n_rows, 32),
        triton.cdiv(n_cols, 32),
    )

    addmm_kernel[grid](input.data_ptr(), mat1.data_ptr(), mat2.data_ptr(), out.data_ptr(),
                        n_rows, n_cols, k, alpha, beta, BLOCK_SIZE=32)

    return out
