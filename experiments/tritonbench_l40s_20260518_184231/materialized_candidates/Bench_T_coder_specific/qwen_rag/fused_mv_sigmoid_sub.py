import triton
import triton.language as tl

@triton.jit
def fused_mv_sigmoid_sub_kernel(
    input_ptr, vec_ptr, other_ptr, output_ptr,
    n, m, alpha, block_size: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row_start = pid * block_size
    row_end = min(row_start + block_size, n)

    # Load data into shared memory
    z_shared = tl.zeros((block_size, m), dtype=tl.float32)
    s_shared = tl.zeros((block_size, m), dtype=tl.float32)
    y_shared = tl.zeros((block_size, m), dtype=tl.float32)

    for i in range(m):
        input_value = tl.load(input_ptr + (row_start + tl.arange(block_size)) * m + i)
        vec_value = tl.load(vec_ptr + i)
        z_shared[:, i] += input_value * vec_value

    # Perform sigmoid activation
    for i in range(m):
        z_value = z_shared[:, i]
        s_shared[:, i] = 1.0 / (1.0 + tl.exp(-z_value))

    # Perform subtraction
    for i in range(m):
        other_value = alpha * tl.load(other_ptr + i)
        y_shared[:, i] = s_shared[:, i] - other_value

    # Write results back to global memory
    for i in range(m):
        tl.store(output_ptr + (row_start + tl.arange(block_size)) * m + i, y_shared[:, i])
