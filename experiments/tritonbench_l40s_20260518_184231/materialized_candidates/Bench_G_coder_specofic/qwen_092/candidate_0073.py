import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_1pass_kernel(
    data_ptr, weights_ptr, bias_ptr, output_ptr, residual_ptr,
    mean_ptr, var_ptr, residual_output_ptr,
    N, C, BLOCK_N: tl.constexpr, EPS: tl.constexpr,
    USE_RESIDUAL: tl.constexpr, STORE_RESIDUAL_OUTPUT: tl.constexpr,
    RMS_NORM: tl.constexpr, APPLY_BIAS: tl.constexpr
):
    # Calculate the row index
    row_idx = tl.program_id(0)
    row_start = row_idx * C
    row_end = row_start + C

    # Initialize variables
    sum_data = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sum_data_sq = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Compute the mean and variance
    for i in range(row_start, row_end):
        data = tl.load(data_ptr + i)
        sum_data[tl.arange(BLOCK_N)] += data
        sum_data_sq[tl.arange(BLOCK_N)] += data * data

    # Reduce to compute global sum and sum of squares
    sum_data = tl.sum(sum_data, axis=0)
    sum_data_sq = tl.sum(sum_data_sq, axis=0)
    tl.store(mean_ptr + row_idx, sum_data / C)
    tl.store(var_ptr + row_idx, sum_data_sq / C - (sum_data / C) ** 2 + EPS)

    # Normalize the data
    norm_data = (data - mean_ptr[row_idx]) / tl.sqrt(var_ptr[row_idx])

    # Apply weights and bias
    for i in range(row_start, row_end):
        output_ptr[i] = norm_data[i - row_start] * weights_ptr[i % C]
        if APPLY_BIAS:
            output_ptr[i] += bias_ptr[i % C]

    # Store residual output if needed
    if STORE_RESIDUAL_OUTPUT:
        for i in range(row_start, row_end):
            residual_output_ptr[i] = output_ptr[i]

    # Store residual if needed
    if USE_RESIDUAL:
        for i in range(row_start, row_end):
            residual_ptr[i] = output_ptr[i]
