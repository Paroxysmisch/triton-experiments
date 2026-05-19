import triton
import triton.language as tl

@triton.jit
def std_kernel(
    x_ptr,
    means_ptr,
    variances_ptr,
    n_elements_ptr,
    num_rows,
    num_cols,
    block_size_x,
    block_size_y,
    correction,
):
    pid = tl.program_id(axis=0)
    row_start = pid * block_size_x
    col_start = pid * block_size_y

    # Load data
    x = tl.load(x_ptr + (row_start * num_cols + col_start))

    # Calculate mean and variance
    mean = tl.zeros((block_size_x,), dtype=tl.float32)
    variance = tl.zeros((block_size_x,), dtype=tl.float32)

    for i in range(block_size_x):
        mean[i] += x[i]
        variance[i] += x[i] * x[i]

    mean /= block_size_x
    variance /= block_size_x
    variance -= mean * mean

    # Adjust for correction
    adjusted_variance = variance / max(0, block_size_x - correction)

    # Store results
    tl.store(means_ptr + row_start, mean)
    tl.store(variances_ptr + row_start, adjusted_variance)

# Register the kernel
triton.register_kernel(std_kernel, "std_kernel", signature="(F, F, F, F, I32, I32, I32, I32, I32)->()")
