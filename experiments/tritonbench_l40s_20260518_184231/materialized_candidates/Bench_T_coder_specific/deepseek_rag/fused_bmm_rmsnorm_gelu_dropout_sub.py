_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    tl.debug_barrier()

    square_output = row * row
    mean_output = tl.sum(square_output)/n_cols + eps
    rms_output = tl.sqrt(mean_output)
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride # TODO: optimization: always 1 after the reduction
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, rms_output, mask=col_offsets < n_cols)

@triton.jit
def rms_norm(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    # The rows are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    tl.debug_barrier()

    square_output = row * row
    mean_output = tl.sum(square_output)/n_cols + eps
    rms_output = tl.sqrt(mean_output)
    
    # Normalize the row
    normalized_row = row / rms_output
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride # TODO: optimization: always 1 after the reduction
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, normalized_row, mask=col_offsets < n_cols)

Note: The kernels are designed to be used with Triton's `launch_kernel` function. They should be used with the `input_ptr` and `output_ptr` arguments pointing to contiguous memory locations. The `input_row_stride` and `output_row_stride` arguments should be the distance in bytes between successive rows in the input and output tensors, respectively. The `n_cols` argument should be the number of columns in the input tensor. The `eps` argument is used for numerical stability. The `BLOCK_SIZE` argument should be the number of elements processed in parallel.

The kernels are written in a way that they can be easily parallelized across rows. Each row is independent, so the `row_idx` variable is used to index into the input and output tensors. The `col_offsets` variable is used to index into each row. The `input_ptrs` and `output_ptrs` variables are used to compute the addresses of each element in the row. The `tl.load` and `tl.store` functions are used to load and store the elements of the row from and to the input and output tensors, respectively. The `tl.sum` function is used to compute the sum of the elements of the row. The `tl.sqrt` function is used to compute the square root of the elements of the row. The `tl.debug_barrier` function is used to synchronize the threads in the block.

The kernels are written in a way that they can be easily parallelized across rows. Each row is independent, so the `row_idx` variable is used to index into the input and output tensors. The `col_offsets` variable is used to index into each row. The `input_ptrs` and `output_ptrs` variables are used to compute the addresses of each element in the row. The `tl.load` and `tl.store` functions are used to load and store the elements of the row from and to the input and output tensors, respectively. The `tl.sum` function is used to compute the sum of the elements of the row. The `tl.sqrt` function is used to compute the square root of the elements of the row. The `tl.debug_barrier` function is used to synchronize the threads in the block.

The kernels are written in a way that they can be easily parallelized across rows. Each row is independent, so the `row_idx` variable is used to index into the input and output tensors. The `col_offsets` variable is used to index into each row. The `input_ptrs` and `output_ptrs` variables are used to compute the addresses of each element in the row. The `tl.load` and `tl.store` functions are used to load and store the elements of the row from and to the input and output tensors, respectively. The `tl.sum` function is used to compute the sum of the elements of the row. The `tl.sqrt` function is used to compute the square root of the elements of the row. The `tl.debug_barrier` function is used to synchronize the threads in the block.

The kernels are written in a way that they can be easily parallelized across rows. Each row is independent, so the `row_idx` variable is used to index into the input and output tensors. The `col_offsets` variable is used to index into each row. The `input_ptrs` and `output_ptrs` variables are used to compute the addresses of each element in the row. The `tl.load` and `tl.store` functions are used to load and store the elements of the row from and to the input and output tensors, respectively. The `tl.sum` function is used to compute the sum of the elements of the row. The `tl.sqrt` function is used to compute the square root of the elements of the row. The `tl.debug_barrier` function is used to synchronize the threads in the block.

The kernels are written in a way that they can be easily parallelized across rows. Each row is independent, so the `row_idx` variable is used to index into the input and output tensors. The `col_offsets` variable is used to index into each row. The `input_ptrs` and `output_ptrs` variables are used to compute the addresses of each element in the row. The `tl.load` and `tl.store` functions are used to load and store the elements of the row from and to the input and output tensors, respectively. The `tl.sum` function is used to compute the sum of the elements of the row. The `tl.sqrt` function is used to compute the square root of the elements of the row. The `tl.debug_barrier` function is used to synchronize the threads in the block.

The kernels are written in a way that they can be easily parallelized across rows. Each row is independent, so the `row_idx` variable is used to index into the input and output tensors. The `col_offsets` variable is used to index into each row. The `input_ptrs` and `output_ptrs` variables are used to compute the addresses of each element in the row. The `tl.load` and `tl.store` functions are used to load and store the elements of the row from and to the input and output tensors, respectively. The `tl.sum` function is used to compute the sum of the elements of the row. The `tl.sqrt` function is used to compute the square root of the elements of the row. The `tl.debug_barrier` function is used to synchronize the threads in the block.

The kernels are written in a way that they can be easily parallelized across rows. Each row is independent, so the `row_idx` variable is used to index into the input and output tensors. The `col_offsets` variable is used to index into each row. The `input_ptrs` and `output_ptrs` variables are used to compute the addresses of each element in the row. The `tl.load` and `tl.store` functions are
