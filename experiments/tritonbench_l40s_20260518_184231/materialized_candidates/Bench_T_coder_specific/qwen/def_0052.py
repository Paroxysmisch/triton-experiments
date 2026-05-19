import triton
import triton.language as tl

@triton.jit
def sum_std_kernel(
    X_ptr,
    Y_sum_ptr,
    Y_std_ptr,
    N,
    stride_x,
    stride_y_sum,
    stride_y_std,
    dim_size,
    num_elements_per_block,
    grid_idx,
):
    # Compute the index within the block
    idx = tl.program_id(axis=0)
    
    # Compute the base index in the input tensor
    base_idx = idx * num_elements_per_block
    
    # Initialize sum and squared sum
    sum_val = tl.zeros([], dtype=tl.float32)
    squared_sum = tl.zeros([], dtype=tl.float32)
    
    # Loop over the elements in the block
    for i in range(num_elements_per_block):
        global_idx = base_idx + i
        if global_idx < N:
            x = tl.load(X_ptr + global_idx * stride_x)
            sum_val += x
            squared_sum += x * x
    
    # Reduce across blocks
    sum_val = tl.all_reduce(sum_val, group=triton.sync_grid_dim_1d())
    squared_sum = tl.all_reduce(squared_sum, group=triton.sync_grid_dim_1d())
    
    # Compute mean and variance
    mean = sum_val / dim_size
    variance = (squared_sum / dim_size - mean * mean) / (dim_size - 1)
    
    # Store results
    tl.store(Y_sum_ptr + grid_idx * stride_y_sum, sum_val)
    tl.store(Y_std_ptr + grid_idx * stride_y_std, tl.sqrt(variance))
